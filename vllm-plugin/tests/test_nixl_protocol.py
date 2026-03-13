"""Tests for the NIXL protobuf protocol.

These tests verify the wire format and round-trip behavior WITHOUT
requiring vLLM, NIXL, PyTorch, or real GPUs. All external deps are
either mocked or not needed.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from vllm_layercast.nixl_agent import VramNixlAgent

import pytest

from vllm_layercast.proto import (
    Error,
    IpcRequest,
    IpcResponse,
    ModelLoaded,
    ModelUnloaded,
    Ok,
    PeerNixlMd,
    Prepared,
    PrepareModel,
    TensorInfo,
    decode_response,
    encode,
)


# PeerNixlMd tests (structured proto, replaces compound msgpack)


class TestPeerNixlMd:
    """Verify the structured PeerNixlMd proto round-trips correctly."""

    def test_roundtrip(self) -> None:
        nixl_md = b"\xde\xad\xbe\xef" * 100
        tensors = [
            TensorInfo(
                name="q_proj.weight",
                addr=139637976039424,
                size=134217728,
                device_id=0,
                dtype="torch.bfloat16",
                checksum=0,
            ),
            TensorInfo(
                name="k_proj.weight",
                addr=139637976173568,
                size=33554432,
                device_id=0,
                dtype="torch.bfloat16",
                checksum=0,
            ),
        ]
        peer = PeerNixlMd(agent_name="test", nixl_md=nixl_md, tensors=tensors)
        wire = bytes(peer)
        decoded = PeerNixlMd().parse(wire)

        assert decoded.nixl_md == nixl_md
        assert len(decoded.tensors) == 2
        assert decoded.tensors[0].name == "q_proj.weight"
        assert decoded.tensors[0].addr == 139637976039424
        assert decoded.tensors[1].size == 33554432

    def test_empty_manifest(self) -> None:
        peer = PeerNixlMd(agent_name="test", nixl_md=b"\xca\xfe\xba\xbe", tensors=[])
        wire = bytes(peer)
        decoded = PeerNixlMd().parse(wire)
        assert decoded.tensors == []
        assert decoded.nixl_md == b"\xca\xfe\xba\xbe"

    def test_large_manifest(self) -> None:
        """Simulate a 500-parameter model (typical for 70B)."""
        tensors = [
            TensorInfo(
                name=f"model.layers.{i}.self_attn.q_proj.weight",
                addr=0x7F00_0000_0000 + i * 0x0800_0000,
                size=134217728,
                device_id=0,
                dtype="torch.bfloat16",
                checksum=0,
            )
            for i in range(500)
        ]
        peer = PeerNixlMd(agent_name="test", nixl_md=b"\x00" * 8192, tensors=tensors)
        wire = bytes(peer)
        decoded = PeerNixlMd().parse(wire)

        assert len(decoded.tensors) == 500
        manifest = {t.name: t for t in decoded.tensors}
        assert (
            manifest["model.layers.0.self_attn.q_proj.weight"].addr == 0x7F00_0000_0000
        )
        assert manifest["model.layers.499.self_attn.q_proj.weight"].addr == (
            0x7F00_0000_0000 + 499 * 0x0800_0000
        )

    def test_manifest_lookup_by_name(self) -> None:
        """Verify the dict-based lookup pattern used in _try_nixl_load."""
        tensors = [
            TensorInfo(name="embed.weight", addr=100, size=1000, device_id=0),
            TensorInfo(name="lm_head.weight", addr=200, size=2000, device_id=1),
        ]
        peer = PeerNixlMd(agent_name="test", nixl_md=b"", tensors=tensors)
        wire = bytes(peer)
        decoded = PeerNixlMd().parse(wire)

        remote_manifest = {t.name: t for t in decoded.tensors}
        assert remote_manifest["embed.weight"].addr == 100
        assert remote_manifest["embed.weight"].size == 1000
        assert remote_manifest["lm_head.weight"].device_id == 1


# IPC wire format tests (state machine messages)


class TestIpcWireFormat:
    """Verify messages survive the IPC protobuf encoding."""

    def test_prepare_model_encode(self) -> None:
        msg = IpcRequest(
            prepare_model=PrepareModel(
                model_id="Qwen/Qwen2.5-3B", revision="main", tp_rank=0
            )
        )
        wire = encode(msg)

        (length,) = struct.unpack(">I", wire[:4])
        assert length == len(wire) - 4

        decoded = IpcRequest().parse(wire[4:])
        assert decoded.is_set("prepare_model")
        assert decoded.prepare_model.model_id == "Qwen/Qwen2.5-3B"
        assert decoded.prepare_model.revision == "main"
        assert decoded.prepare_model.tp_rank == 0

    def test_model_loaded_with_structured_metadata(self) -> None:
        """ModelLoaded carries nixl_md and tensors as separate proto fields."""
        msg = IpcRequest(
            model_loaded=ModelLoaded(
                agent_name="layercast-worker-0-rank0",
                nixl_md=b"\xca\xfe\xba\xbe",
                tensors=[
                    TensorInfo(
                        name="layers.0.weight",
                        addr=0x7F00_0000_0000,
                        size=4096,
                        device_id=0,
                    ),
                    TensorInfo(
                        name="layers.0.bias",
                        addr=0x7F00_0000_1000,
                        size=128,
                        device_id=0,
                    ),
                ],
                model_id="Qwen/Qwen2.5-3B",
                files=["model-00001-of-00002.safetensors"],
                tp_rank=0,
            )
        )

        wire = encode(msg)
        (length,) = struct.unpack(">I", wire[:4])
        assert length == len(wire) - 4

        decoded = IpcRequest().parse(wire[4:])
        ml = decoded.model_loaded
        assert ml.agent_name == "layercast-worker-0-rank0"
        assert ml.nixl_md == b"\xca\xfe\xba\xbe"
        assert len(ml.tensors) == 2
        assert ml.tensors[0].name == "layers.0.weight"
        assert ml.tensors[1].size == 128

    def test_model_unloaded_encode(self) -> None:
        msg = IpcRequest(model_unloaded=ModelUnloaded(agent_name="worker-0"))
        wire = encode(msg)
        decoded = IpcRequest().parse(wire[4:])
        assert decoded.model_unloaded.agent_name == "worker-0"

    def test_prepared_response_with_peers(self) -> None:
        """Prepared response carries structured peer metadata."""
        resp = IpcResponse(
            prepared=Prepared(
                files=["shard-001.safetensors"],
                peers=[
                    PeerNixlMd(
                        agent_name="remote-agent",
                        nixl_md=b"\xca\xfe\xba\xbe",
                        tensors=[
                            TensorInfo(
                                name="layers.0.weight",
                                addr=0x7F00_0000_0000,
                                size=4096,
                                device_id=0,
                            ),
                        ],
                    )
                ],
            )
        )
        wire = bytes(resp)
        decoded = decode_response(wire)
        assert decoded.prepared.files == ["shard-001.safetensors"]
        assert len(decoded.prepared.peers) == 1
        assert decoded.prepared.peers[0].nixl_md == b"\xca\xfe\xba\xbe"
        assert len(decoded.prepared.peers[0].tensors) == 1

    def test_prepared_response_with_weight_map(self) -> None:
        """Prepared response carries weight_map for multi-shard models."""
        resp = IpcResponse(
            prepared=Prepared(
                files=[
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
                peers=[],
                weight_map={
                    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
                },
            )
        )
        wire = bytes(resp)
        decoded = decode_response(wire)
        assert len(decoded.prepared.weight_map) == 2
        assert (
            decoded.prepared.weight_map["model.layers.0.self_attn.q_proj.weight"]
            == "model-00001-of-00002.safetensors"
        )

    def test_prepared_response_empty_weight_map(self) -> None:
        """Single-shard models have empty weight_map (proto3 default)."""
        resp = IpcResponse(
            prepared=Prepared(
                files=["model.safetensors"],
                peers=[],
            )
        )
        wire = bytes(resp)
        decoded = decode_response(wire)
        assert decoded.prepared.weight_map == {}

    def test_model_id_no_revision(self) -> None:
        """Verify model_id uses bare repo_id (no @revision suffix)."""
        msg = IpcRequest(
            model_loaded=ModelLoaded(
                agent_name="a",
                nixl_md=b"",
                model_id="meta-llama/Llama-3-70B",
                files=["model.safetensors"],
                tp_rank=0,
            )
        )
        wire = encode(msg)
        decoded = IpcRequest().parse(wire[4:])
        assert "@" not in decoded.model_loaded.model_id
        assert decoded.model_loaded.model_id == "meta-llama/Llama-3-70B"


# Protobuf message validation tests


class TestProtoMessages:
    """Verify protobuf messages serialize and parse correctly."""

    def test_request_oneof_discrimination(self) -> None:
        """Oneof field correctly identifies the active variant."""
        msg = IpcRequest(
            prepare_model=PrepareModel(model_id="org/model", revision="main", tp_rank=0)
        )
        assert msg.is_set("prepare_model")

    def test_model_loaded_bytes_roundtrip(self) -> None:
        """bytes fields survive encode -> parse round-trip."""
        original_nixl_md = b"\xde\xad" * 50
        msg = IpcRequest(
            model_loaded=ModelLoaded(
                agent_name="test",
                nixl_md=original_nixl_md,
                model_id="org/model",
                files=["shard.safetensors"],
                tp_rank=0,
            )
        )
        wire = bytes(msg)
        restored = IpcRequest().parse(wire)
        assert restored.model_loaded.nixl_md == original_nixl_md

    def test_response_oneof_variants(self) -> None:
        """decode_response correctly dispatches all response variants."""
        for resp, expected_field in [
            (
                IpcResponse(
                    prepared=Prepared(
                        files=["shard.safetensors"],
                        peers=[],
                    )
                ),
                "prepared",
            ),
            (IpcResponse(ok=Ok()), "ok"),
            (IpcResponse(error=Error(message="boom")), "error"),
        ]:
            wire = bytes(resp)
            result = decode_response(wire)
            assert result.is_set(expected_field)

    def test_peer_nixl_md_in_prepared(self) -> None:
        """PeerNixlMd nested inside Prepared validates correctly."""
        resp = IpcResponse(
            prepared=Prepared(
                files=["shard.safetensors"],
                peers=[
                    PeerNixlMd(
                        agent_name="peer-0",
                        nixl_md=b"\x01\x02\x03",
                    ),
                    PeerNixlMd(
                        agent_name="peer-1",
                        nixl_md=b"\x04\x05\x06",
                    ),
                ],
            )
        )
        wire = bytes(resp)
        result = decode_response(wire)
        p = result.prepared
        assert len(p.peers) == 2
        assert p.peers[0].agent_name == "peer-0"
        assert p.peers[1].nixl_md == b"\x04\x05\x06"


# TensorInfo tests


class TestTensorInfo:
    """Test TensorInfo proto used for weight manifests."""

    def test_manifest_populated_on_register(self) -> None:
        """Simulates what VramNixlAgent.register_vram produces."""
        tensors_data = [
            ("layer.0.weight", 0x7F0000, 4096, 0),
            ("layer.0.bias", 0x7F1000, 1024, 0),
        ]
        tensor_infos = [
            TensorInfo(name=n, addr=a, size=s, device_id=d)
            for n, a, s, d in tensors_data
        ]

        assert len(tensor_infos) == 2
        by_name = {t.name: t for t in tensor_infos}
        assert by_name["layer.0.weight"].addr == 0x7F0000
        assert by_name["layer.0.weight"].size == 4096
        assert by_name["layer.0.bias"].size == 1024
        assert by_name["layer.0.bias"].device_id == 0

    def test_manifest_survives_proto_roundtrip(self) -> None:
        tensor_infos = [
            TensorInfo(name="attn.q", addr=0xAAAA0000, size=4096, device_id=0),
            TensorInfo(name="attn.k", addr=0xBBBB0000, size=4096, device_id=0),
            TensorInfo(name="attn.v", addr=0xCCCC0000, size=4096, device_id=1),
        ]
        peer = PeerNixlMd(agent_name="test", nixl_md=b"\x00" * 64, tensors=tensor_infos)
        wire = bytes(peer)
        decoded = PeerNixlMd().parse(wire)
        remote = {t.name: t for t in decoded.tensors}

        assert remote["attn.q"].addr == 0xAAAA0000
        assert remote["attn.k"].addr == 0xBBBB0000
        assert remote["attn.v"].addr == 0xCCCC0000
        assert remote["attn.v"].device_id == 1

    def test_manifest_with_checksums_roundtrip(self) -> None:
        """Checksum field survives protobuf round-trip."""
        tensor_infos = [
            TensorInfo(
                name="layer.0.weight",
                addr=0x7F0000,
                size=4096,
                device_id=0,
                dtype="torch.bfloat16",
                checksum=0xDEADBEEFCAFEBABE,
            ),
            TensorInfo(
                name="layer.0.bias",
                addr=0x7F1000,
                size=128,
                device_id=0,
                dtype="torch.float32",
                checksum=0x1234567890ABCDEF,
            ),
        ]
        peer = PeerNixlMd(agent_name="test", nixl_md=b"\x00" * 64, tensors=tensor_infos)
        wire = bytes(peer)
        decoded = PeerNixlMd().parse(wire)
        remote = {t.name: t for t in decoded.tensors}

        assert remote["layer.0.weight"].checksum == 0xDEADBEEFCAFEBABE
        assert remote["layer.0.bias"].checksum == 0x1234567890ABCDEF

    def test_manifest_without_checksum_default_zero(self) -> None:
        """Proto3 default: checksum=0 means unset."""
        t = TensorInfo(name="w", addr=100, size=4096, device_id=0)
        wire = bytes(t)
        decoded = TensorInfo().parse(wire)
        assert decoded.checksum == 0

    def test_missing_tensor_detection(self) -> None:
        remote_tensors = [
            TensorInfo(name="layer.0.weight", addr=100, size=4096, device_id=0),
        ]
        remote_manifest = {t.name: t for t in remote_tensors}

        local_names = ["layer.0.weight", "layer.0.bias", "layer.1.weight"]
        missing = [n for n in local_names if n not in remote_manifest]

        assert len(missing) == 2
        assert "layer.0.bias" in missing
        assert "layer.1.weight" in missing


# wait_transfer tests


class TestWaitTransfer:
    """Test VramNixlAgent.wait_transfer polling, error detection, and timeout."""

    def _make_agent(self) -> "VramNixlAgent":
        """Build a VramNixlAgent with a mocked NIXL backend."""
        import vllm_layercast.nixl_agent as mod

        orig_agent = mod._nixl_agent
        orig_config = mod._nixl_agent_config

        mock_nixl = MagicMock()
        mock_config_cls = MagicMock()
        mod._nixl_agent = mock_nixl
        mod._nixl_agent_config = mock_config_cls

        try:
            from vllm_layercast.nixl_agent import VramNixlAgent

            agent = VramNixlAgent("test-agent")
        finally:
            mod._nixl_agent = orig_agent
            mod._nixl_agent_config = orig_config

        return agent

    def test_immediate_done(self) -> None:
        agent = self._make_agent()
        agent._agent.check_xfer_state.return_value = "DONE"
        agent.wait_transfer("fake-handle")
        agent._agent.check_xfer_state.assert_called_once_with("fake-handle")

    def test_done_after_polling(self) -> None:
        agent = self._make_agent()
        agent._agent.check_xfer_state.side_effect = ["PROC", "PROC", "PROC", "DONE"]
        agent.wait_transfer("h")
        assert agent._agent.check_xfer_state.call_count == 4

    def test_error_raises_runtime_error(self) -> None:
        """NIXL returns 'ERR' (not 'ERROR') on transfer failure."""
        agent = self._make_agent()
        agent._agent.check_xfer_state.side_effect = ["PROC", "ERR"]
        with pytest.raises(RuntimeError, match="NIXL transfer failed"):
            agent.wait_transfer("h")

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import vllm_layercast.nixl_agent as mod

        monkeypatch.setattr(mod, "_TRANSFER_TIMEOUT_S", 0.0)
        agent = self._make_agent()
        agent._agent.check_xfer_state.return_value = "PROC"
        with pytest.raises(TimeoutError, match="did not complete"):
            agent.wait_transfer("h")
