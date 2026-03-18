"""Tests for the LayercastBackend (gRPC transport).

Uses a real gRPC server in-process to verify actual round-trips.
The fake server speaks the layercast protobuf service definition.
"""

from __future__ import annotations

from collections.abc import Generator
from concurrent import futures

import grpc
import pytest

from vllm_layercast.backend_rust import LayercastBackend
from vllm_layercast.pb.layercast import (
    ModelLoadedRequest,
    ModelLoadedResponse,
    ModelUnloadedRequest,
    ModelUnloadedResponse,
    Ok,
    PeerNixlMd,
    Prepared,
    PrepareModelRequest,
    PrepareModelResponse,
    TensorInfo,
)


class FakeLayercastServicer(grpc.GenericRpcHandler):
    """In-process gRPC handler that records requests and returns canned responses.

    Implements GenericRpcHandler so we can register without codegen.
    """

    def __init__(self) -> None:
        self.prepare_responses: list[PrepareModelResponse] = []
        self.loaded_responses: list[ModelLoadedResponse] = []
        self.unloaded_responses: list[ModelUnloadedResponse] = []
        self.received_prepare: list[PrepareModelRequest] = []
        self.received_loaded: list[ModelLoadedRequest] = []
        self.received_unloaded: list[ModelUnloadedRequest] = []

    def service(
        self, handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler | None:
        method: str = handler_call_details.method  # type: ignore[assignment]
        if method == "/layercast.LayercastService/PrepareModel":
            return grpc.unary_unary_rpc_method_handler(self._handle_prepare)
        if method == "/layercast.LayercastService/ModelLoaded":
            return grpc.unary_unary_rpc_method_handler(self._handle_loaded)
        if method == "/layercast.LayercastService/ModelUnloaded":
            return grpc.unary_unary_rpc_method_handler(self._handle_unloaded)
        return None

    def _handle_prepare(
        self, request_bytes: bytes, context: grpc.ServicerContext
    ) -> bytes:
        req = PrepareModelRequest().parse(request_bytes)
        self.received_prepare.append(req)
        if self.prepare_responses:
            resp = self.prepare_responses.pop(0)
        else:
            resp = PrepareModelResponse(prepared=Prepared())
        return bytes(resp)

    def _handle_loaded(
        self, request_bytes: bytes, context: grpc.ServicerContext
    ) -> bytes:
        req = ModelLoadedRequest().parse(request_bytes)
        self.received_loaded.append(req)
        if self.loaded_responses:
            resp = self.loaded_responses.pop(0)
        else:
            resp = ModelLoadedResponse(ok=Ok())
        return bytes(resp)

    def _handle_unloaded(
        self, request_bytes: bytes, context: grpc.ServicerContext
    ) -> bytes:
        req = ModelUnloadedRequest().parse(request_bytes)
        self.received_unloaded.append(req)
        if self.unloaded_responses:
            resp = self.unloaded_responses.pop(0)
        else:
            resp = ModelUnloadedResponse(ok=Ok())
        return bytes(resp)


def _start_fake_server(
    servicer: FakeLayercastServicer,
) -> tuple[grpc.Server, int]:
    """Start a real gRPC server on a random port. Returns (server, port)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    server.add_generic_rpc_handlers([servicer])
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


class TestLayercastBackend:
    """Test LayercastBackend with a real in-process gRPC server."""

    @pytest.fixture
    def servicer(self) -> FakeLayercastServicer:
        return FakeLayercastServicer()

    @pytest.fixture
    def server_addr(
        self, servicer: FakeLayercastServicer
    ) -> Generator[tuple[str, grpc.Server]]:
        server, port = _start_fake_server(servicer)
        yield f"127.0.0.1:{port}", server
        server.stop(grace=0)

    @pytest.mark.asyncio
    async def test_start_connects(self, server_addr: tuple[str, grpc.Server]) -> None:
        """start() opens the gRPC channel."""
        addr, _ = server_addr
        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        assert backend._client is not None
        await backend.stop()

    @pytest.mark.asyncio
    async def test_stop_disconnects(self, server_addr: tuple[str, grpc.Server]) -> None:
        """stop() closes the channel and resets state."""
        addr, _ = server_addr
        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        await backend.stop()
        assert backend._client is None

    @pytest.mark.asyncio
    async def test_ensure_client_before_start_raises(self) -> None:
        """_ensure_client raises RuntimeError before start() is called."""
        backend = LayercastBackend(server_addr="127.0.0.1:0")
        with pytest.raises(RuntimeError, match="not connected"):
            backend._ensure_client()

    @pytest.mark.asyncio
    async def test_prepare_model(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
    ) -> None:
        """prepare_model returns Prepared from server."""
        addr, _ = server_addr
        servicer.prepare_responses.append(
            PrepareModelResponse(
                prepared=Prepared(
                    files=["shard-001.safetensors", "shard-002.safetensors"],
                    peers=[
                        PeerNixlMd(agent_name="peer-0", nixl_md=b"\xde\xad"),
                        PeerNixlMd(agent_name="peer-1", nixl_md=b"\xbe\xef"),
                    ],
                )
            )
        )

        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        prepared = await backend.prepare_model("org/model", "main", 0)
        assert prepared.files == ["shard-001.safetensors", "shard-002.safetensors"]
        assert len(prepared.peers) == 2
        assert prepared.peers[0].agent_name == "peer-0"
        assert prepared.peers[0].nixl_md == b"\xde\xad"
        assert prepared.peers[1].agent_name == "peer-1"
        await backend.stop()

    @pytest.mark.asyncio
    async def test_prepare_model_with_weight_map(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
    ) -> None:
        """prepare_model returns Prepared with weight_map."""
        addr, _ = server_addr
        servicer.prepare_responses.append(
            PrepareModelResponse(
                prepared=Prepared(
                    files=[
                        "model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors",
                    ],
                    peers=[],
                    weight_map={
                        "layers.0.weight": "model-00001-of-00002.safetensors",
                        "layers.1.weight": "model-00002-of-00002.safetensors",
                    },
                )
            )
        )

        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        prepared = await backend.prepare_model("org/model", "main", 0)
        assert len(prepared.weight_map) == 2
        assert (
            prepared.weight_map["layers.0.weight"] == "model-00001-of-00002.safetensors"
        )
        await backend.stop()

    @pytest.mark.asyncio
    async def test_model_loaded(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
    ) -> None:
        """model_loaded sends to server without error."""
        addr, _ = server_addr
        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        await backend.model_loaded(
            agent_name="test-agent",
            nixl_md=b"\x00\x01\x02",
            tensors=[],
            model_id="org/model",
            files=["shard.safetensors"],
            tp_rank=0,
        )
        await backend.stop()

        assert len(servicer.received_loaded) == 1
        req = servicer.received_loaded[0]
        assert req.loaded.agent_name == "test-agent"
        assert req.loaded.nixl_md == b"\x00\x01\x02"
        assert req.loaded.model_id == "org/model"

    @pytest.mark.asyncio
    async def test_model_unloaded(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
    ) -> None:
        """model_unloaded sends to server without error."""
        addr, _ = server_addr
        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        await backend.model_unloaded("test-agent")
        await backend.stop()

        assert len(servicer.received_unloaded) == 1
        assert servicer.received_unloaded[0].unloaded.agent_name == "test-agent"

    @pytest.mark.asyncio
    async def test_state_machine_happy_path(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
    ) -> None:
        """Full lifecycle: prepare -> loaded -> unloaded."""
        addr, _ = server_addr
        servicer.prepare_responses.append(
            PrepareModelResponse(
                prepared=Prepared(
                    files=["shard.safetensors"],
                    peers=[],
                )
            )
        )

        backend = LayercastBackend(server_addr=addr)
        await backend.start()

        prepared = await backend.prepare_model("org/model", "main", 0)
        assert prepared.files == ["shard.safetensors"]
        assert prepared.peers == []

        await backend.model_loaded(
            agent_name="agent-0",
            nixl_md=b"\xca\xfe",
            tensors=[],
            model_id="org/model",
            files=["shard.safetensors"],
            tp_rank=0,
        )
        await backend.model_unloaded("agent-0")
        await backend.stop()

        assert len(servicer.received_prepare) == 1
        assert len(servicer.received_loaded) == 1
        assert len(servicer.received_unloaded) == 1

    @pytest.mark.asyncio
    async def test_prepare_model_sends_pod_identity(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Wrapper messages include pod_name and pod_ip from env."""
        addr, _ = server_addr
        monkeypatch.setenv("POD_NAME", "vllm-seed-0")
        monkeypatch.setenv("POD_IP", "10.0.0.42")

        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        await backend.prepare_model("org/model", "main", 0)
        await backend.stop()

        assert len(servicer.received_prepare) == 1
        req = servicer.received_prepare[0]
        assert req.pod_name == "vllm-seed-0"
        assert req.pod_ip == "10.0.0.42"
        assert req.prepare.model_id == "org/model"
        assert req.prepare.revision == "main"
        assert req.prepare.tp_rank == 0

    @pytest.mark.asyncio
    async def test_model_loaded_with_tensors(
        self,
        servicer: FakeLayercastServicer,
        server_addr: tuple[str, grpc.Server],
    ) -> None:
        """model_loaded sends tensor metadata correctly."""
        addr, _ = server_addr
        tensors = [
            TensorInfo(
                name="layers.0.weight",
                addr=0x7F0000000000,
                size=1024 * 1024,
                device_id=0,
                dtype="torch.bfloat16",
                checksum=12345,
            ),
        ]

        backend = LayercastBackend(server_addr=addr)
        await backend.start()
        await backend.model_loaded(
            agent_name="agent-0",
            nixl_md=b"\xde\xad\xbe\xef",
            tensors=tensors,
            model_id="org/model",
            files=["shard.safetensors"],
            tp_rank=0,
        )
        await backend.stop()

        req = servicer.received_loaded[0]
        assert len(req.loaded.tensors) == 1
        t = req.loaded.tensors[0]
        assert t.name == "layers.0.weight"
        assert t.addr == 0x7F0000000000
        assert t.size == 1024 * 1024
        assert t.dtype == "torch.bfloat16"
        assert t.checksum == 12345
