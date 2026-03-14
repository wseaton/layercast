"""Layercast backend: delegates to the central metadata server via gRPC."""

from __future__ import annotations

from vllm_layercast.grpc_client import GrpcClient
from vllm_layercast.log import get_logger
from vllm_layercast.proto import Prepared, TensorInfo

log = get_logger("vllm_layercast.backend")


class LayercastBackend:
    """Backend that talks to the layercast metadata server over gRPC.

    Each backend instance maintains one persistent gRPC channel. The server
    tracks per-pod state machines with automatic cleanup via Pod watcher.
    """

    def __init__(self, server_addr: str) -> None:
        self._server_addr = server_addr
        self._client: GrpcClient | None = None

    async def start(self) -> None:
        self._client = GrpcClient(server_addr=self._server_addr)
        await self._client.connect()
        log.info("started", server=self._server_addr)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _ensure_client(self) -> GrpcClient:
        if self._client is None:
            raise RuntimeError("LayercastBackend not connected. Call start() first.")
        return self._client

    async def prepare_model(
        self,
        model_id: str,
        revision: str,
        tp_rank: int,
        peer_discovery_timeout_s: int | None = None,
    ) -> Prepared:
        client = self._ensure_client()
        return await client.prepare_model(
            model_id, revision, tp_rank, peer_discovery_timeout_s
        )

    async def model_loaded(
        self,
        agent_name: str,
        nixl_md: bytes,
        tensors: list[TensorInfo],
        model_id: str,
        files: list[str],
        tp_rank: int,
    ) -> None:
        client = self._ensure_client()
        await client.model_loaded(
            agent_name=agent_name,
            nixl_md=nixl_md,
            tensors=tensors,
            model_id=model_id,
            files=files,
            tp_rank=tp_rank,
        )

    async def model_unloaded(self, agent_name: str) -> None:
        client = self._ensure_client()
        await client.model_unloaded(agent_name)


# Keep the old name as an alias during migration
RustSidecarBackend = LayercastBackend
