"""Layercast model loader for vLLM.

Implements the BaseModelLoader interface with a cascade:
  1. GPU VRAM direct via NIXL GPUDirect RDMA (peer -> local GPU)
  2. Fall through to vLLM's default HF loader (NFS PVC or HF download)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import json
import os
import time
import urllib.request
from typing import TYPE_CHECKING

import msgpack

from vllm_layercast.log import get_logger

log = get_logger("vllm_layercast.loader")

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise RuntimeError("PyTorch is required for vllm-layercast")

try:
    from vllm.config import LoadConfig, ModelConfig, VllmConfig
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
except ImportError:
    if not TYPE_CHECKING:
        raise RuntimeError(
            "vLLM is required for the layercast loader. "
            "Install it or run within a vLLM environment."
        )

from vllm_layercast.backend_rust import RustSidecarBackend  # noqa: E402
from vllm_layercast.checksum import is_enabled as _checksum_enabled  # noqa: E402
from vllm_layercast.checksum import verify_checksums  # noqa: E402
from vllm_layercast.nixl_agent import VramNixlAgent, _COALESCE_THRESHOLD  # noqa: E402
from vllm_layercast.proto import PeerNixlMd  # noqa: E402

# Keep NIXL agents alive for the process lifetime so remote peers can
# RDMA-read. Without this, the agent (and its registered memory)
# could be garbage collected when the model loader is no longer referenced.
_NIXL_AGENTS: dict[int, VramNixlAgent] = {}  # keyed by tp_rank

# Keep the backend + event loop alive for the process lifetime.
# Without this, Python GC collects the model loader after load_model()
# returns, closing the Unix socket. The sidecar interprets the EOF as a
# crash and auto-unadvertises, nuking the CRD entry before any consumer
# can discover us.
_BACKEND_KEEPALIVE: tuple[asyncio.AbstractEventLoop, RustSidecarBackend] | None = None


class LayercastModelLoader(BaseModelLoader):
    """P2P model loader.

    Cascade: NIXL -> vLLM default (NFS PVC / HF download).
    """

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        log.info("init", pid=os.getpid())
        self._nixl_agent: VramNixlAgent | None = None
        self._backend: RustSidecarBackend | None = None
        self._backend_loop: asyncio.AbstractEventLoop | None = None
        self._weight_files_cache: dict[str, list[str]] = {}
        # The fallback needs load_format="auto" (safetensors/HF), not "layercast"
        fallback_config = dataclasses.replace(load_config, load_format="auto")
        self._fallback = DefaultModelLoader(fallback_config)

    def download_model(self, model_config: ModelConfig) -> None:
        """Pre-download model files.

        With NFS PVC handling weight distribution, this just delegates
        to the default loader which reads from the shared filesystem.
        """
        self._fallback.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights with the NIXL -> default cascade."""
        t_start = time.monotonic()
        log.info("load_weights_start", model=model_config.model)
        tp_rank = _get_tp_rank()
        repo_id = model_config.model
        revision = model_config.revision or "main"

        # Single batched call for file list + peers
        weight_files, peers = self._prepare_model(repo_id, revision, tp_rank)
        if not weight_files:
            weight_files = self._get_weight_files(model_config)
        if not weight_files:
            log.info("no_weight_files", fallback="default_loader")
            self._fallback.load_weights(model, model_config)
            return

        loaded = False

        # first, NIXL GPUDirect RDMA from peer VRAM
        if self._try_nixl_load(
            model, repo_id, revision, weight_files, tp_rank, peers=peers
        ):
            loaded = True
            elapsed = time.monotonic() - t_start
            log.info("weights_loaded", method="nixl", elapsed_s=elapsed, model=repo_id)

        # fall back to vLLM default, either local fs or HF download
        if not loaded:
            log.info("fallback_to_default")
            self._fallback.load_weights(model, model_config)
            elapsed = time.monotonic() - t_start
            log.info(
                "weights_loaded", method="hf_default", elapsed_s=elapsed, model=repo_id
            )

        # register VRAM with the daemon so we can serve peers
        self._publish_vram(model, repo_id, revision, weight_files, tp_rank)

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        """Initialize the model, load weights, and run post-processing.

        Tries NIXL GPUDirect first, with full fallback to DefaultModelLoader
        which handles init + weights + post-processing.

        We check for peers BEFORE initializing the model to avoid double-init
        (vLLM's attention layer registry doesn't allow duplicate layer names).
        """
        t_start = time.monotonic()
        tp_rank = _get_tp_rank()
        repo_id = model_config.model
        revision = model_config.revision or "main"

        # Single batched call: file list + peer discovery + metadata fetch.
        # Falls back to local dir scan / HF API if backend is unavailable.
        weight_files, peers = self._prepare_model(repo_id, revision, tp_rank)
        if not weight_files:
            weight_files = self._get_weight_files(model_config)

        if peers:
            # Overlap NIXL agent + UCX backend init (~3s) with model init (~16s).
            # Both are independent: model init builds the graph on meta device,
            # while NIXL creates UCX workers and endpoints.
            agent_future = concurrent.futures.ThreadPoolExecutor(1).submit(
                self._ensure_nixl_agent, tp_rank
            )
            model = initialize_model(vllm_config, prefix=prefix)
            # Collect the agent (will be instant if model init took longer)
            try:
                agent_future.result(timeout=30)
            except Exception as exc:
                log.error("nixl_agent_init_failed", error=str(exc))
            if self._try_nixl_load(
                model,
                repo_id,
                revision,
                weight_files,
                tp_rank,
                peers=peers,
            ):
                elapsed = time.monotonic() - t_start
                log.info(
                    "model_loaded", method="nixl", elapsed_s=elapsed, model=repo_id
                )
                target_device = torch.device(vllm_config.device_config.device)
                process_weights_after_loading(model, model_config, target_device)
                # Move computed buffers (e.g. rotary cos_sin_cache) from
                # CPU to GPU. NIXL only transfers parameters, not buffers.
                model = model.to(target_device)
                self._publish_vram(model, repo_id, revision, weight_files, tp_rank)
                return model
            # NIXL load failed after init. This is a fatal error since
            # we can't call initialize_model twice (attention layer registry).
            log.error("nixl_load_failed_after_init")
            raise RuntimeError("NIXL GPUDirect load failed after model initialization")

        # No P2P sources: delegate entirely to DefaultModelLoader
        log.info("no_p2p_sources", fallback="default_loader")
        model = self._fallback.load_model(vllm_config, model_config, prefix=prefix)
        elapsed = time.monotonic() - t_start
        log.info("model_loaded", method="hf_default", elapsed_s=elapsed, model=repo_id)

        # Publish VRAM so future peers can NIXL-load from us
        if weight_files:
            self._publish_vram(model, repo_id, revision, weight_files, tp_rank)
        return model

    def _try_nixl_load(
        self,
        model: nn.Module,
        repo_id: str,
        revision: str,
        weight_files: list[str],
        tp_rank: int,
        peers: list[PeerNixlMd] | None = None,
    ) -> bool:
        """Zero-copy NIXL GPUDirect RDMA weight transfer from peer VRAM.

        After initialize_model(), params live on meta device (shape/dtype known
        but no storage). We materialize each param to CUDA, then NIXL READ from
        the seed's VRAM directly into our parameter storage. The result is that
        model.parameters() point at buffers filled by RDMA with no intermediate
        copies.

        The compound metadata format (msgpack):
          {"nixl_md": <bytes>, "tensors": [{"name", "addr", "size", "device_id"}, ...]}
        """
        if peers is None:
            _, peers = self._prepare_model(repo_id, revision, tp_rank)

        if not peers:
            log.debug("nixl_no_peers", model=repo_id, rank=tp_rank)
            return False

        peer = peers[0]
        log.info(
            "nixl_peer_found", agent=peer.agent_name, metadata_bytes=len(peer.metadata)
        )

        try:
            agent = self._ensure_nixl_agent(tp_rank)
        except Exception as exc:
            log.warning("nixl_agent_init_failed", error=str(exc))
            return False

        try:
            compound = msgpack.unpackb(peer.metadata, raw=False)
            nixl_md = compound["nixl_md"]
            remote_manifest = {t["name"]: t for t in compound["tensors"]}
        except (KeyError, TypeError, msgpack.UnpackException):
            log.warning("nixl_bad_metadata")
            return False

        if not remote_manifest:
            log.warning("nixl_empty_manifest")
            return False

        peer_name = agent.load_peer_metadata(nixl_md)
        log.info(
            "nixl_peer_loaded", peer=peer_name, remote_tensors=len(remote_manifest)
        )

        # Materialize meta params to CUDA and match against remote manifest.
        # Use the remote manifest's dtype (not the meta-device param dtype,
        # which defaults to float32 before weight loading).
        t_xfer = time.monotonic()
        device = f"cuda:{torch.cuda.current_device()}"
        local_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name not in remote_manifest:
                continue
            remote_info = remote_manifest[name]
            dtype = _parse_torch_dtype(remote_info.get("dtype"), param.dtype)
            cuda_buf = torch.empty(param.shape, dtype=dtype, device=device)
            param.data = cuda_buf
            local_tensors[name] = cuda_buf

        if not local_tensors:
            log.warning("nixl_no_matching_params")
            return False

        log.info(
            "nixl_materialized",
            matched=len(local_tensors),
            remote=len(remote_manifest),
        )

        # Split tensors into large (individual transfers) and small (coalesced).
        # Small tensors (<1MB, e.g. layer norms, biases) get packed into a
        # contiguous CUDA buffer and transferred in one NIXL operation to
        # avoid per-transfer overhead.
        large_tensors: dict[str, torch.Tensor] = {}
        small_entries: list[tuple[str, torch.Tensor]] = []
        for name, tensor in local_tensors.items():
            nbytes = tensor.nelement() * tensor.element_size()
            if nbytes < _COALESCE_THRESHOLD:
                small_entries.append((name, tensor))
            else:
                large_tensors[name] = tensor

        # Only register large tensors with NIXL (they receive data directly).
        # Small tensors receive via the coalesce buffer, so skip their
        # individual registration to save ~3s.
        agent.register_vram(large_tensors)

        handles: list[tuple[str, object]] = []

        # Coalesced transfer for small tensors
        if small_entries:
            total_small = sum(t.nelement() * t.element_size() for _, t in small_entries)
            coalesce_buf = torch.empty(total_small, dtype=torch.uint8, device=device)
            agent.register_vram({"__coalesced__": coalesce_buf})

            remote_regions: list[tuple[int, int, int]] = []
            offsets: list[int] = []
            scatter_plan: list[tuple[str, int, int]] = []  # (name, offset, nbytes)
            offset = 0
            for name, tensor in small_entries:
                remote = remote_manifest[name]
                nbytes = remote["size"]
                remote_regions.append((remote["addr"], nbytes, remote["device_id"]))
                offsets.append(offset)
                scatter_plan.append((name, offset, nbytes))
                offset += nbytes

            try:
                handle = agent.read_coalesced_from_peer(
                    coalesce_buf,
                    remote_regions,
                    offsets,
                    peer_name,
                )
                handles.append(("__coalesced__", handle))
            except Exception as exc:
                log.warning("nixl_coalesced_read_failed", error=str(exc))
                return False

            log.info(
                "nixl_coalesced",
                tensors=len(small_entries),
                total_kb=total_small / 1024,
            )

        # Individual transfers for large tensors
        for name, local_tensor in large_tensors.items():
            remote = remote_manifest[name]
            try:
                handle = agent.read_from_peer(
                    local_tensor=local_tensor,
                    remote_addr=remote["addr"],
                    remote_len=remote["size"],
                    peer_name=peer_name,
                    remote_device_id=remote["device_id"],
                )
                handles.append((name, handle))
            except Exception as exc:
                log.warning("nixl_read_failed", tensor=name, error=str(exc))
                return False

        # Wait for all transfers
        for name, handle in handles:
            try:
                agent.wait_transfer(handle)
            except RuntimeError as exc:
                log.warning("nixl_transfer_failed", tensor=name, error=str(exc))
                return False

        # Build expected checksums from remote manifest (may be empty for
        # old sources that don't publish checksums, in which case we skip).
        expected_checksums: dict[str, int] = {}
        if _checksum_enabled():
            expected_checksums = {
                name: info["checksum"]
                for name, info in remote_manifest.items()
                if info.get("checksum") is not None
            }

        t_checksum = time.monotonic()

        # Verify large tensors (already in their final CUDA buffers)
        if expected_checksums and large_tensors:
            mismatched = verify_checksums(large_tensors, expected_checksums)
            if mismatched:
                log.error(
                    "checksum_verification_failed",
                    phase="large",
                    mismatched=mismatched,
                    peer=peer_name,
                )
                return False

        # Scatter coalesced buffer back into individual param tensors
        if small_entries:
            for name, buf_offset, nbytes in scatter_plan:
                src = coalesce_buf[buf_offset : buf_offset + nbytes]
                dst = local_tensors[name]
                dst.data.copy_(src.view(dst.dtype).reshape(dst.shape))
            del coalesce_buf

        # Verify small tensors AFTER scatter (checksum was computed on the
        # source's original tensor, not the coalesced uint8 layout)
        if expected_checksums and small_entries:
            small_tensor_dict = {
                name: local_tensors[name] for name, _, _ in scatter_plan
            }
            mismatched = verify_checksums(small_tensor_dict, expected_checksums)
            if mismatched:
                log.error(
                    "checksum_verification_failed",
                    phase="small",
                    mismatched=mismatched,
                    peer=peer_name,
                )
                return False

        checksum_elapsed = time.monotonic() - t_checksum

        # Log verification result when checksums were checked
        if expected_checksums:
            log.info(
                "checksum_verification",
                verified=len(expected_checksums),
                mismatched=0,
                elapsed_s=round(checksum_elapsed, 3),
            )

        xfer_elapsed = time.monotonic() - t_xfer
        total_bytes = sum(
            t.nelement() * t.element_size() for t in local_tensors.values()
        )
        gbps = (total_bytes / 1e9) / xfer_elapsed if xfer_elapsed > 0 else 0.0
        log.info(
            "nixl_transfer_complete",
            elapsed_s=xfer_elapsed,
            total_bytes=total_bytes,
            gbps=gbps,
            tensors=len(local_tensors),
            large=len(large_tensors),
            coalesced=len(small_entries),
            peer=peer_name,
        )
        return True

    def _publish_vram(
        self,
        model: nn.Module,
        repo_id: str,
        revision: str,
        weight_files: list[str],
        tp_rank: int,
    ) -> None:
        """Register loaded VRAM tensors with NIXL and publish metadata.

        Packs both the NIXL agent metadata and the tensor address manifest
        into a compound msgpack payload. The backend treats it as opaque bytes.

        If tensors are already registered (e.g. from _try_nixl_load), we
        skip re-registration to avoid the ~5s cost.
        """
        try:
            agent = self._ensure_nixl_agent(tp_rank)
        except Exception as exc:
            log.warning("nixl_agent_init_failed_publish", error=str(exc))
            return

        gpu_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.is_cuda:
                gpu_tensors[name] = param.data

        if not gpu_tensors:
            log.debug("no_cuda_tensors_to_publish")
            return

        log.info("publishing_vram", tensor_count=len(gpu_tensors))
        try:
            agent.register_vram(gpu_tensors)
        except Exception as exc:
            log.warning("nixl_register_vram_failed", error=str(exc))
            return

        # Pack NIXL metadata + tensor address manifest into compound payload.
        # Peers unpack this to get both the NIXL blob (for agent import) and
        # the address map (for constructing remote transfer descriptors).
        try:
            compound = msgpack.packb(
                {
                    "nixl_md": agent.get_metadata(),
                    "tensors": agent.get_weight_manifest(),
                },
                use_bin_type=True,
            )
        except Exception as exc:
            log.warning("nixl_metadata_pack_failed", error=str(exc))
            return

        log.info("publishing_nixl_metadata", bytes=len(compound))
        loop = self._get_backend_loop()
        try:
            backend = loop.run_until_complete(self._get_backend())
            loop.run_until_complete(
                backend.model_loaded(
                    agent_name=agent.name,
                    metadata=compound,
                    model_id=repo_id,
                    files=weight_files,
                    tp_rank=tp_rank,
                )
            )
            log.info(
                "vram_published",
                tensors=len(gpu_tensors),
                rank=tp_rank,
            )
            # Pin the backend + loop so GC can't close the IPC socket.
            # The sidecar treats EOF as a crash and unadvertises immediately.
            global _BACKEND_KEEPALIVE  # noqa: PLW0603
            _BACKEND_KEEPALIVE = (loop, backend)
        except Exception as exc:
            log.warning("publish_nixl_metadata_failed", error=str(exc))

    # Helpers

    def _get_backend_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the dedicated event loop for backend IPC.

        The DaemonClient's socket reader/writer are bound to a specific event
        loop. vLLM may run its own event loop in the main thread later, so we
        keep a dedicated loop to avoid "Future attached to a different loop".
        """
        if self._backend_loop is None or self._backend_loop.is_closed():
            self._backend_loop = asyncio.new_event_loop()
        return self._backend_loop

    def _prepare_model(
        self, repo_id: str, revision: str, tp_rank: int
    ) -> tuple[list[str], list[PeerNixlMd]]:
        """Call backend.prepare_model() to get files + peers in one round-trip.

        Falls back to local directory scan / HF API if the backend is unavailable.
        """
        loop = self._get_backend_loop()
        try:
            backend = loop.run_until_complete(self._get_backend())
            files, peers = loop.run_until_complete(
                backend.prepare_model(repo_id, revision, tp_rank)
            )
            if files:
                log.info(
                    "prepare_model_ok",
                    files=len(files),
                    peers=len(peers),
                    model=repo_id,
                )
            return files, peers
        except Exception as exc:
            log.debug("prepare_model_failed", error=str(exc))
        return [], []

    def _get_weight_files(self, model_config: ModelConfig) -> list[str]:
        """Cached weight file discovery.

        Tries three strategies:
        1. Local directory (NFS, shared FS, etc.)
        2. Query the HF model info API directly
        3. Fetch the safetensors index file

        Note: the backend's prepare_model() is the primary path (called from
        load_model/load_weights), but this provides a fallback for cases where
        the backend isn't available yet.
        """
        key = f"{model_config.model}@{model_config.revision or 'main'}"
        if key in self._weight_files_cache:
            return self._weight_files_cache[key]

        model_name = model_config.model
        if not model_name:
            return []

        revision = model_config.revision or "main"
        files: list[str] = []

        # Strategy 0: local directory (NFS, shared FS, etc.)
        if os.path.isdir(model_name):
            import glob as _glob

            local_files = sorted(
                os.path.basename(f)
                for f in _glob.glob(os.path.join(model_name, "*.safetensors"))
            )
            if local_files:
                log.info(
                    "weight_files_discovered",
                    source="local_dir",
                    count=len(local_files),
                )
                self._weight_files_cache[key] = local_files
                return local_files

        # Strategy 1: query HF model info API directly
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        try:
            files = _discover_via_hf_api(model_name, revision, hf_endpoint)
            if files:
                log.info("weight_files_discovered", source="hf_api", count=len(files))
        except Exception as exc:
            log.debug("hf_api_unavailable", error=str(exc))

        # Strategy 2: fetch the safetensors index file
        if not files:
            try:
                files = _discover_via_index(model_name, revision, hf_endpoint)
                if files:
                    log.info(
                        "weight_files_discovered", source="index_file", count=len(files)
                    )
            except Exception as exc:
                log.debug("index_file_unavailable", error=str(exc))

        if not files:
            log.warning("no_weight_files_found", model=model_name)

        self._weight_files_cache[key] = files
        return files

    def _ensure_nixl_agent(self, tp_rank: int) -> VramNixlAgent:
        if self._nixl_agent is None:
            if tp_rank in _NIXL_AGENTS:
                self._nixl_agent = _NIXL_AGENTS[tp_rank]
            else:
                hostname = os.environ.get("HOSTNAME", "unknown")
                name = f"layercast-{hostname}-rank{tp_rank}"
                agent = VramNixlAgent(name)
                _NIXL_AGENTS[tp_rank] = agent
                self._nixl_agent = agent
        return self._nixl_agent

    async def _get_backend(self) -> RustSidecarBackend:
        """Lazily create and connect the backend."""
        if self._backend is None:
            socket_path = os.environ.get(
                "LAYERCAST_SOCKET", "/var/run/layercast/daemon.sock"
            )
            self._backend = RustSidecarBackend(socket_path=socket_path)
            await self._backend.start()
        return self._backend


def _get_tp_rank() -> int:
    """Get tensor parallel rank, defaulting to 0 if not in a TP group."""
    try:
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _discover_via_hf_api(model_name: str, revision: str, hf_endpoint: str) -> list[str]:
    """Query HF model info API for the file listing."""
    api_url = f"{hf_endpoint}/api/models/{model_name}/revision/{revision}"
    req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.loads(resp.read())

    siblings = info.get("siblings", [])
    return sorted(
        s["rfilename"]
        for s in siblings
        if isinstance(s, dict) and s.get("rfilename", "").endswith(".safetensors")
    )


def _discover_via_index(model_name: str, revision: str, hf_endpoint: str) -> list[str]:
    """Fetch the safetensors index JSON and extract shard filenames."""
    index_url = (
        f"{hf_endpoint}/{model_name}/resolve/{revision}/model.safetensors.index.json"
    )
    req = urllib.request.Request(index_url)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    with urllib.request.urlopen(req, timeout=30) as resp:
        index = json.loads(resp.read())

    weight_map = index.get("weight_map", {})
    return sorted(set(weight_map.values()))


def _parse_torch_dtype(dtype_str: str | None, fallback: "torch.dtype") -> "torch.dtype":
    """Convert a dtype string like 'torch.bfloat16' to a torch.dtype.

    Falls back to the provided default if the string is missing or invalid.
    """
    if dtype_str is None:
        return fallback
    # Handle both "torch.bfloat16" and "bfloat16" formats
    name = dtype_str.removeprefix("torch.")
    result = getattr(torch, name, None)
    if result is not None and isinstance(result, torch.dtype):
        return result
    return fallback
