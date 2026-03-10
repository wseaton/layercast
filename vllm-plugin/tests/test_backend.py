"""Tests for the RustSidecarBackend.

Uses real Unix sockets with a fake daemon to verify actual IPC round-trips.
"""

from __future__ import annotations

import asyncio
import os
import struct
import tempfile

import msgpack
import pytest

from vllm_layercast.backend_rust import RustSidecarBackend


# Fake daemon: a real Unix socket server that speaks the wire protocol


async def _fake_daemon(
    socket_path: str, responses: list[dict]
) -> asyncio.AbstractServer:
    """Start a Unix socket server that replies with pre-canned responses.

    Each client connection gets the next response from the list.
    Returns the server so the caller can close it.
    """
    response_iter = iter(responses)

    async def handle_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                # Read length-prefixed request
                length_bytes = await reader.readexactly(4)
                (length,) = struct.unpack(">I", length_bytes)
                await reader.readexactly(length)  # consume request body

                # Send next canned response
                try:
                    resp = next(response_iter)
                except StopIteration:
                    resp = {"type": "ok"}

                payload = msgpack.packb(resp, use_bin_type=True)
                writer.write(struct.pack(">I", len(payload)) + payload)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    return server


# RustSidecarBackend integration tests (real Unix socket IPC)


class TestRustSidecarBackend:
    """Test RustSidecarBackend with a real Unix socket and fake daemon."""

    @pytest.fixture
    def socket_path(self, tmp_path: object) -> str:
        """Create a temp path for the Unix socket."""
        d = tempfile.mkdtemp()
        return os.path.join(d, "test.sock")

    @pytest.mark.asyncio
    async def test_start_connects(self, socket_path: str) -> None:
        """start() connects to the daemon socket."""
        server = await _fake_daemon(socket_path, [])
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()
            assert backend._client is not None
            assert backend._client._writer is not None
            await backend.stop()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_stop_disconnects(self, socket_path: str) -> None:
        """stop() closes the connection and resets state."""
        server = await _fake_daemon(socket_path, [])
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()
            await backend.stop()
            assert backend._client is None
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_start_no_socket_raises(self) -> None:
        """start() raises FileNotFoundError if the socket doesn't exist."""
        backend = RustSidecarBackend(socket_path="/tmp/nonexistent_layercast_test.sock")
        with pytest.raises(FileNotFoundError):
            await backend.start()

    @pytest.mark.asyncio
    async def test_ensure_client_before_start_raises(self) -> None:
        """_ensure_client raises RuntimeError before start() is called."""
        backend = RustSidecarBackend(socket_path="/tmp/whatever.sock")
        with pytest.raises(RuntimeError, match="not connected"):
            backend._ensure_client()

    @pytest.mark.asyncio
    async def test_prepare_model(self, socket_path: str) -> None:
        """prepare_model returns files and peers from daemon."""
        server = await _fake_daemon(
            socket_path,
            [
                {
                    "type": "prepared",
                    "files": ["shard-001.safetensors", "shard-002.safetensors"],
                    "peers": [
                        {"agent_name": "peer-0", "metadata": b"\xde\xad"},
                        {"agent_name": "peer-1", "metadata": b"\xbe\xef"},
                    ],
                },
            ],
        )
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()
            files, peers = await backend.prepare_model("org/model", "main", 0)
            assert files == ["shard-001.safetensors", "shard-002.safetensors"]
            assert len(peers) == 2
            assert peers[0].agent_name == "peer-0"
            assert peers[0].metadata == b"\xde\xad"
            assert peers[1].agent_name == "peer-1"
            await backend.stop()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_model_loaded(self, socket_path: str) -> None:
        """model_loaded sends to daemon without error."""
        server = await _fake_daemon(
            socket_path,
            [
                {"type": "ok"},
            ],
        )
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()
            await backend.model_loaded(
                agent_name="test-agent",
                metadata=b"\x00\x01\x02",
                model_id="org/model",
                files=["shard.safetensors"],
                tp_rank=0,
            )
            await backend.stop()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_model_unloaded(self, socket_path: str) -> None:
        """model_unloaded sends to daemon without error."""
        server = await _fake_daemon(
            socket_path,
            [
                {"type": "ok"},
            ],
        )
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()
            await backend.model_unloaded("test-agent")
            await backend.stop()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_prepare_model_error_raises(self, socket_path: str) -> None:
        """prepare_model raises RuntimeError on error response."""
        server = await _fake_daemon(
            socket_path,
            [
                {"type": "error", "message": "HF API unreachable"},
            ],
        )
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()
            with pytest.raises(RuntimeError, match="PrepareModel failed"):
                await backend.prepare_model("org/model", "main", 0)
            await backend.stop()
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_state_machine_happy_path(self, socket_path: str) -> None:
        """Full lifecycle: prepare -> loaded -> unloaded."""
        server = await _fake_daemon(
            socket_path,
            [
                {
                    "type": "prepared",
                    "files": ["shard.safetensors"],
                    "peers": [],
                },
                {"type": "ok"},  # model_loaded
                {"type": "ok"},  # model_unloaded
            ],
        )
        try:
            backend = RustSidecarBackend(socket_path=socket_path)
            await backend.start()

            files, peers = await backend.prepare_model("org/model", "main", 0)
            assert files == ["shard.safetensors"]
            assert peers == []

            await backend.model_loaded(
                agent_name="agent-0",
                metadata=b"\xca\xfe",
                model_id="org/model",
                files=["shard.safetensors"],
                tp_rank=0,
            )

            await backend.model_unloaded("agent-0")

            await backend.stop()
        finally:
            server.close()
            await server.wait_closed()
