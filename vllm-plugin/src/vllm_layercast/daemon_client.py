"""Async Unix socket client for the layercast daemon."""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

from vllm_layercast.log import get_logger
from vllm_layercast.proto import (
    Error,
    ModelLoaded,
    ModelUnloaded,
    Ok,
    PeerNixlMd,
    Prepared,
    PrepareModel,
    RequestMessage,
    decode_response,
    encode,
)

log = get_logger("vllm_layercast.daemon")

DEFAULT_SOCKET_PATH = "/var/run/layercast/daemon.sock"


class DaemonClient:
    """Talks to the layercast Rust daemon over a Unix domain socket.

    Wire format: 4-byte big-endian length prefix + msgpack payload.
    Speaks the state machine protocol (PrepareModel -> ModelLoaded -> ModelUnloaded).
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH) -> None:
        self.socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Open the Unix socket connection to the daemon."""
        path = Path(self.socket_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Daemon socket not found at {self.socket_path}. "
                "Is the layercast daemon running?"
            )
        self._reader, self._writer = await asyncio.open_unix_connection(
            self.socket_path
        )
        log.info("connected", socket=self.socket_path)

    async def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    # State machine RPC methods

    async def prepare_model(
        self,
        model_id: str,
        revision: str,
        tp_rank: int,
    ) -> tuple[list[str], list[PeerNixlMd]]:
        """Ask daemon to list files + discover NIXL peers.

        Returns (files, peers). The daemon does file listing and peer metadata
        fetch in one batched operation.
        """
        msg = PrepareModel(model_id=model_id, revision=revision, tp_rank=tp_rank)
        resp = await self._roundtrip(msg)
        if isinstance(resp, Error):
            raise RuntimeError(f"PrepareModel failed: {resp.message}")
        if not isinstance(resp, Prepared):
            raise TypeError(f"Expected Prepared, got {type(resp).__name__}")
        return resp.files, resp.peers

    async def model_loaded(
        self,
        agent_name: str,
        metadata: bytes,
        model_id: str,
        files: list[str],
        tp_rank: int,
    ) -> None:
        """Tell daemon model is loaded: store NIXL metadata + advertise via CRD."""
        msg = ModelLoaded(
            agent_name=agent_name,
            metadata=metadata,
            model_id=model_id,
            files=files,
            tp_rank=tp_rank,
        )
        resp = await self._roundtrip(msg)
        if isinstance(resp, Error):
            raise RuntimeError(f"ModelLoaded failed: {resp.message}")

    async def model_unloaded(self, agent_name: str) -> None:
        """Tell daemon to unadvertise + remove metadata."""
        msg = ModelUnloaded(agent_name=agent_name)
        resp = await self._roundtrip(msg)
        if isinstance(resp, Error):
            raise RuntimeError(f"ModelUnloaded failed: {resp.message}")

    # Internal helpers

    def _ensure_reader(self) -> asyncio.StreamReader:
        if self._reader is None:
            raise RuntimeError("Not connected to daemon. Call connect() first.")
        return self._reader

    def _ensure_writer(self) -> asyncio.StreamWriter:
        if self._writer is None:
            raise RuntimeError("Not connected to daemon. Call connect() first.")
        return self._writer

    async def _send(self, msg: RequestMessage) -> None:
        writer = self._ensure_writer()
        writer.write(encode(msg))
        await writer.drain()

    async def _recv(self) -> Prepared | Ok | Error | dict[str, object]:
        reader = self._ensure_reader()
        length_bytes = await reader.readexactly(4)
        (length,) = struct.unpack(">I", length_bytes)
        max_message_size = 64 * 1024 * 1024  # 64 MB, matches Rust daemon limit
        if length > max_message_size:
            raise ValueError(
                f"Response too large: {length} bytes (max {max_message_size})"
            )
        data = await reader.readexactly(length)
        return decode_response(data)

    async def _roundtrip(
        self, msg: RequestMessage
    ) -> Prepared | Ok | Error | dict[str, object]:
        await self._send(msg)
        return await self._recv()
