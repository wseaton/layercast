"""VRAM NIXL agent wrapper for GPU-direct RDMA transfers."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from vllm_layercast.checksum import is_enabled as _checksum_enabled
from vllm_layercast.checksum import tensor_wyhash
from vllm_layercast.log import get_logger

log = get_logger("vllm_layercast.nixl")

# Threshold below which tensors are coalesced into a single contiguous
# buffer for transfer. Avoids per-transfer overhead for tiny norm/bias
# tensors. Configurable via env var.
_COALESCE_THRESHOLD = int(
    os.environ.get("LAYERCAST_COALESCE_THRESHOLD", str(1 * 1024 * 1024))
)

# Number of NIXL UCX worker threads. More threads = more pipeline
# parallelism for concurrent transfers. 0 = NIXL default (single worker).
_NUM_THREADS = int(os.environ.get("LAYERCAST_NIXL_NUM_THREADS", "4"))

# InfiniBand Service Level for weight transfer traffic. Separates our
# bulk, high-throughput transfers onto a different virtual lane from
# latency-sensitive inference traffic (KV cache, etc. on SL=0).
# Set to empty string to disable.
_IB_SERVICE_LEVEL = os.environ.get("LAYERCAST_IB_SL", "1")

# Maximum seconds to wait for a single NIXL transfer before giving up.
# Weight transfers for large models (70B+) can take 30-60s over RDMA.
_TRANSFER_TIMEOUT_S = float(os.environ.get("LAYERCAST_TRANSFER_TIMEOUT", "120"))

if TYPE_CHECKING:
    import torch

_torch: Any = None
try:
    import torch as _torch_import

    _torch = _torch_import
except ImportError:
    pass

_nixl_agent: Any = None
_nixl_agent_config: Any = None
try:
    from nixl._api import nixl_agent as _nixl_agent_import
    from nixl._api import nixl_agent_config as _nixl_agent_config_import

    _nixl_agent = _nixl_agent_import
    _nixl_agent_config = _nixl_agent_config_import
except ImportError:
    pass


class VramNixlAgent:
    """Thin wrapper around NIXL for registering GPU VRAM and doing transfers.

    Each vLLM worker process (one per TP rank) creates its own agent.
    """

    def __init__(self, name: str) -> None:
        if _nixl_agent is None or _nixl_agent_config is None:
            raise RuntimeError(
                "nixl package is not installed. Install it with: pip install nixl"
            )
        # Build UCX engine_config string for QoS and tuning knobs
        engine_cfg_parts: list[str] = []
        if _IB_SERVICE_LEVEL:
            engine_cfg_parts.append(f"IB_SL={_IB_SERVICE_LEVEL}")
        engine_cfg = ",".join(engine_cfg_parts)

        log.info(
            "creating_agent",
            name=name,
            backend="UCX",
            num_threads=_NUM_THREADS,
            engine_config=engine_cfg,
        )
        # Don't auto-init backends via config so we can pass engine_config
        config = _nixl_agent_config(backends=[])
        self._agent = _nixl_agent(name, config)

        ucx_init: dict[str, str] = {}
        if _NUM_THREADS > 0:
            ucx_init["num_threads"] = str(_NUM_THREADS)
        if engine_cfg:
            ucx_init["engine_config"] = engine_cfg
        self._agent.create_backend("UCX", ucx_init)
        self._name = name
        self._registered_descs: list[object] = []
        self._registered_names: set[str] = set()
        self._weight_manifest: list[dict[str, int | str | None]] = []
        log.info("agent_ready", name=name)

    @property
    def name(self) -> str:
        return self._name

    def register_vram(self, tensors: dict[str, "torch.Tensor"]) -> None:
        """Register GPU tensors with NIXL for RDMA transfers.

        Each tensor is registered as a VRAM segment so remote agents can
        read directly from this GPU's memory. Also saves the address manifest
        for sharing with peers so they know where to read from.
        """
        if not tensors:
            return

        new_tensors = {
            name: t for name, t in tensors.items() if name not in self._registered_names
        }
        if not new_tensors:
            log.debug(
                "register_vram_skipped", count=len(tensors), reason="already_registered"
            )
            return

        checksums_enabled = _checksum_enabled()
        descs: list[tuple[int, int, int, str]] = []
        manifest: list[dict[str, int | str | None]] = []
        for param_name, tensor in new_tensors.items():
            addr = tensor.data_ptr()
            nbytes = tensor.nelement() * tensor.element_size()
            device_id = tensor.device.index if tensor.device.index is not None else 0
            descs.append((addr, nbytes, device_id, param_name))
            checksum = tensor_wyhash(tensor) if checksums_enabled else None
            manifest.append(
                {
                    "name": param_name,
                    "addr": addr,
                    "size": nbytes,
                    "device_id": device_id,
                    "dtype": str(tensor.dtype),
                    "checksum": checksum,
                }
            )

        already = len(tensors) - len(new_tensors)
        log.info("registering_vram", new=len(descs), already_registered=already)
        reg = self._agent.register_memory(descs, mem_type="VRAM")
        self._registered_descs.append(reg)
        self._registered_names.update(new_tensors.keys())
        self._weight_manifest.extend(manifest)
        log.info("vram_registered", count=len(descs))

    def register_local_vram(self, tensors: dict[str, "torch.Tensor"]) -> None:
        """Register GPU tensors as local NIXL destinations only.

        Used by the receive path (_issue_and_wait_transfers) to register
        destination buffers for inbound NIXL READs. Does NOT update
        _registered_names or _weight_manifest, so a subsequent register_vram
        call in _publish_vram will re-register these tensors for serving
        with the correct manifest entries and metadata export.
        """
        if not tensors:
            return
        descs: list[tuple[int, int, int, str]] = []
        for name, tensor in tensors.items():
            addr = tensor.data_ptr()
            nbytes = tensor.nelement() * tensor.element_size()
            device_id = tensor.device.index if tensor.device.index is not None else 0
            descs.append((addr, nbytes, device_id, name))
        log.info("registering_local_vram", count=len(descs))
        reg = self._agent.register_memory(descs, mem_type="VRAM")
        self._registered_descs.append(reg)
        log.info("local_vram_registered", count=len(descs))

    def get_metadata(self) -> bytes:
        """Export this agent's metadata for sharing with remote agents."""
        md = self._agent.get_agent_metadata()
        log.debug("metadata_exported", bytes=len(md))
        return md

    def get_weight_manifest(self) -> list[dict[str, int | str | None]]:
        """Return the tensor address manifest from the last register_vram() call.

        Each entry: {"name": str, "addr": int, "size": int, "device_id": int,
        "dtype": str, "checksum": int | None}.
        Peers need this to know where to issue NIXL reads from.
        """
        return self._weight_manifest

    def load_peer_metadata(self, metadata: bytes) -> str:
        """Import a remote agent's metadata. Returns the remote agent name."""
        log.info("loading_peer_metadata", bytes=len(metadata))
        name = self._agent.add_remote_agent(metadata)
        log.info("peer_loaded", peer=name)
        return name

    def read_from_peer(
        self,
        local_tensor: "torch.Tensor",
        remote_addr: int,
        remote_len: int,
        peer_name: str,
        remote_device_id: int = 0,
    ) -> object:
        """Initiate a NIXL Read from a remote peer's VRAM into a local tensor.

        Returns a transfer handle that should be passed to wait_transfer().
        """
        local_addr = local_tensor.data_ptr()
        local_len = local_tensor.nelement() * local_tensor.element_size()
        local_device = (
            local_tensor.device.index if local_tensor.device.index is not None else 0
        )

        local_descs = self._agent.get_xfer_descs(
            [(local_addr, local_len, local_device)],
            mem_type="VRAM",
        )
        remote_descs = self._agent.get_xfer_descs(
            [(remote_addr, remote_len, remote_device_id)],
            mem_type="VRAM",
        )

        handle = self._agent.initialize_xfer(
            operation="READ",
            local_descs=local_descs,
            remote_descs=remote_descs,
            remote_agent=peer_name,
        )
        self._agent.transfer(handle)
        return handle

    def read_coalesced_from_peer(
        self,
        local_buf: "torch.Tensor",
        remote_regions: list[tuple[int, int, int]],
        offsets: list[int],
        peer_name: str,
    ) -> object:
        """Bulk-read multiple small remote regions into a contiguous local buffer.

        Each remote region is (addr, length, device_id). The offsets list gives
        the byte offset into local_buf where each region's data should land.
        All regions are transferred in a single NIXL operation.

        Returns a transfer handle for wait_transfer().
        """
        local_base = local_buf.data_ptr()
        local_device = (
            local_buf.device.index if local_buf.device.index is not None else 0
        )

        local_descs_raw = [
            (local_base + off, length, local_device)
            for off, (_, length, _) in zip(offsets, remote_regions)
        ]
        local_descs = self._agent.get_xfer_descs(local_descs_raw, mem_type="VRAM")
        remote_descs = self._agent.get_xfer_descs(remote_regions, mem_type="VRAM")

        total_bytes = sum(r[1] for r in remote_regions)
        log.info(
            "coalesced_read",
            regions=len(remote_regions),
            total_bytes=total_bytes,
        )
        handle = self._agent.initialize_xfer(
            operation="READ",
            local_descs=local_descs,
            remote_descs=remote_descs,
            remote_agent=peer_name,
        )
        self._agent.transfer(handle)
        return handle

    def wait_transfer(self, handle: object) -> None:
        """Block until a NIXL transfer completes.

        Polls check_xfer_state in a tight loop (non-blocking call, matches
        NIXL's recommended pattern). Raises on error or timeout.
        """
        deadline = time.monotonic() + _TRANSFER_TIMEOUT_S
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                return
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"NIXL transfer did not complete within {_TRANSFER_TIMEOUT_S}s"
                )

    def close(self) -> None:
        """Release all registered memory and clean up."""
        for desc in self._registered_descs:
            try:
                self._agent.deregister_memory(desc)
            except Exception as exc:
                log.warning("deregister_memory_failed", error=str(exc))
        self._registered_descs.clear()
        self._registered_names.clear()
        self._weight_manifest.clear()
        log.info("agent_closed", name=self._name)
