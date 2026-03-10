"""Rust sidecar backend: delegates to the layercast daemon via Unix socket IPC."""

from __future__ import annotations

from vllm_layercast.daemon_client import DaemonClient
from vllm_layercast.log import get_logger
from vllm_layercast.proto import PeerNixlMd

log = get_logger("vllm_layercast.backend_rust")


class RustSidecarBackend:
    """Backend that talks to the Rust model-mesh sidecar over Unix socket IPC.

    Each backend instance maintains one persistent connection to the daemon.
    The connection tracks state (Connected -> Preparing -> Ready) on the
    daemon side, with automatic cleanup on disconnect (crash protection).
    """

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._client: DaemonClient | None = None

    async def start(self) -> None:
        self._client = DaemonClient(socket_path=self._socket_path)
        await self._client.connect()
        log.info("started", socket=self._socket_path)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _ensure_client(self) -> DaemonClient:
        if self._client is None or self._client._writer is None:
            raise RuntimeError("RustSidecarBackend not connected. Call start() first.")
        return self._client

    async def prepare_model(
        self,
        model_id: str,
        revision: str,
        tp_rank: int,
    ) -> tuple[list[str], list[PeerNixlMd]]:
        client = self._ensure_client()
        return await client.prepare_model(model_id, revision, tp_rank)

    async def model_loaded(
        self,
        agent_name: str,
        metadata: bytes,
        model_id: str,
        files: list[str],
        tp_rank: int,
    ) -> None:
        client = self._ensure_client()
        await client.model_loaded(
            agent_name=agent_name,
            metadata=metadata,
            model_id=model_id,
            files=files,
            tp_rank=tp_rank,
        )

    async def model_unloaded(self, agent_name: str) -> None:
        client = self._ensure_client()
        await client.model_unloaded(agent_name)
