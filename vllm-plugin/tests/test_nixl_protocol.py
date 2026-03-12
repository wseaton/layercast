"""Tests for the NIXL compound metadata protocol.

These tests verify the wire format and round-trip behavior WITHOUT
requiring vLLM, NIXL, PyTorch, or real GPUs. All external deps are
either mocked or not needed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from vllm_layercast.nixl_agent import VramNixlAgent

import msgpack
import pytest

from vllm_layercast.proto import (
    Error,
    ModelLoaded,
    ModelUnloaded,
    Ok,
    Prepared,
    PrepareModel,
    decode_response,
    encode,
)


# Compound payload tests (the core of GPU direct addressing)


def make_compound_payload(
    nixl_md: bytes = b"\xca\xfe\xba\xbe",
    tensors: list[dict] | None = None,
) -> bytes:
    """Build a compound NIXL metadata payload (what _publish_vram sends)."""
    if tensors is None:
        tensors = [
            {
                "name": "layers.0.weight",
                "addr": 0x7F00_0000_0000,
                "size": 4096,
                "device_id": 0,
            },
            {
                "name": "layers.0.bias",
                "addr": 0x7F00_0000_1000,
                "size": 128,
                "device_id": 0,
            },
        ]
    return msgpack.packb({"nixl_md": nixl_md, "tensors": tensors}, use_bin_type=True)


class TestCompoundPayload:
    """Verify the compound metadata format round-trips correctly."""

    def test_roundtrip(self) -> None:
        nixl_md = b"\xde\xad\xbe\xef" * 100
        tensors = [
            {
                "name": "q_proj.weight",
                "addr": 139637976039424,
                "size": 134217728,
                "device_id": 0,
            },
            {
                "name": "k_proj.weight",
                "addr": 139637976173568,
                "size": 33554432,
                "device_id": 0,
            },
        ]
        payload = make_compound_payload(nixl_md, tensors)

        decoded = msgpack.unpackb(payload, raw=False)
        assert decoded["nixl_md"] == nixl_md
        assert len(decoded["tensors"]) == 2
        assert decoded["tensors"][0]["name"] == "q_proj.weight"
        assert decoded["tensors"][0]["addr"] == 139637976039424
        assert decoded["tensors"][1]["size"] == 33554432

    def test_empty_manifest(self) -> None:
        payload = make_compound_payload(tensors=[])
        decoded = msgpack.unpackb(payload, raw=False)
        assert decoded["tensors"] == []
        assert decoded["nixl_md"] == b"\xca\xfe\xba\xbe"

    def test_large_manifest(self) -> None:
        """Simulate a 500-parameter model (typical for 70B)."""
        tensors = [
            {
                "name": f"model.layers.{i}.self_attn.q_proj.weight",
                "addr": 0x7F00_0000_0000 + i * 0x0800_0000,
                "size": 134217728,
                "device_id": 0,
            }
            for i in range(500)
        ]
        payload = make_compound_payload(b"\x00" * 8192, tensors)

        decoded = msgpack.unpackb(payload, raw=False)
        assert len(decoded["tensors"]) == 500
        manifest = {t["name"]: t for t in decoded["tensors"]}
        assert (
            manifest["model.layers.0.self_attn.q_proj.weight"]["addr"]
            == 0x7F00_0000_0000
        )
        assert manifest["model.layers.499.self_attn.q_proj.weight"]["addr"] == (
            0x7F00_0000_0000 + 499 * 0x0800_0000
        )

    def test_manifest_lookup_by_name(self) -> None:
        """Verify the dict-based lookup pattern used in _try_nixl_load."""
        tensors = [
            {"name": "embed.weight", "addr": 100, "size": 1000, "device_id": 0},
            {"name": "lm_head.weight", "addr": 200, "size": 2000, "device_id": 1},
        ]
        payload = make_compound_payload(tensors=tensors)
        decoded = msgpack.unpackb(payload, raw=False)

        remote_manifest = {t["name"]: t for t in decoded["tensors"]}

        assert remote_manifest["embed.weight"]["addr"] == 100
        assert remote_manifest["embed.weight"]["size"] == 1000
        assert remote_manifest["lm_head.weight"]["device_id"] == 1


# IPC wire format tests (state machine messages)


class TestIpcWireFormat:
    """Verify messages survive the IPC msgpack encoding."""

    def test_prepare_model_encode(self) -> None:
        msg = PrepareModel(model_id="Qwen/Qwen2.5-3B", revision="main", tp_rank=0)
        wire = encode(msg)

        (length,) = struct.unpack(">I", wire[:4])
        assert length == len(wire) - 4

        raw = msgpack.unpackb(wire[4:], raw=False)
        assert raw["type"] == "prepare_model"
        assert raw["model_id"] == "Qwen/Qwen2.5-3B"
        assert raw["revision"] == "main"
        assert raw["tp_rank"] == 0

    def test_model_loaded_with_compound_metadata(self) -> None:
        """ModelLoaded carries compound bytes as opaque metadata."""
        compound = make_compound_payload()
        msg = ModelLoaded(
            agent_name="layercast-worker-0-rank0",
            metadata=compound,
            model_id="Qwen/Qwen2.5-3B",
            files=["model-00001-of-00002.safetensors"],
            tp_rank=0,
        )

        wire = encode(msg)
        (length,) = struct.unpack(">I", wire[:4])
        assert length == len(wire) - 4

        raw = msgpack.unpackb(wire[4:], raw=False)
        assert raw["type"] == "model_loaded"
        assert raw["agent_name"] == "layercast-worker-0-rank0"

        inner = msgpack.unpackb(raw["metadata"], raw=False)
        assert "nixl_md" in inner
        assert "tensors" in inner
        assert len(inner["tensors"]) == 2

    def test_model_unloaded_encode(self) -> None:
        msg = ModelUnloaded(agent_name="worker-0")
        wire = encode(msg)
        raw = msgpack.unpackb(wire[4:], raw=False)
        assert raw["type"] == "model_unloaded"
        assert raw["agent_name"] == "worker-0"

    def test_prepared_response_with_compound(self) -> None:
        """Prepared response carries compound bytes back to the client."""
        compound = make_compound_payload()
        wire = msgpack.packb(
            {
                "type": "prepared",
                "files": ["shard-001.safetensors"],
                "peers": [{"agent_name": "remote-agent", "metadata": compound}],
            },
            use_bin_type=True,
        )

        decoded = decode_response(wire)
        assert isinstance(decoded, Prepared)
        assert decoded.files == ["shard-001.safetensors"]
        assert len(decoded.peers) == 1

        inner = msgpack.unpackb(decoded.peers[0].metadata, raw=False)
        assert inner["nixl_md"] == b"\xca\xfe\xba\xbe"
        assert len(inner["tensors"]) == 2

    def test_model_id_no_revision(self) -> None:
        """Verify model_id uses bare repo_id (no @revision suffix)."""
        msg = ModelLoaded(
            agent_name="a",
            metadata=b"",
            model_id="meta-llama/Llama-3-70B",
            files=["model.safetensors"],
            tp_rank=0,
        )
        wire = encode(msg)
        raw = msgpack.unpackb(wire[4:], raw=False)
        assert "@" not in raw["model_id"]
        assert raw["model_id"] == "meta-llama/Llama-3-70B"


# Pydantic model validation tests


class TestPydanticModels:
    """Verify pydantic models validate and serialize correctly."""

    def test_request_type_discriminator(self) -> None:
        """Type field is auto-set and immutable via Literal."""
        msg = PrepareModel(model_id="org/model", revision="main", tp_rank=0)
        assert msg.type == "prepare_model"
        dumped = msg.model_dump()
        assert dumped["type"] == "prepare_model"

    def test_model_loaded_bytes_roundtrip(self) -> None:
        """bytes fields survive model_dump -> msgpack -> unpack -> validate."""
        original_metadata = b"\xde\xad" * 50
        msg = ModelLoaded(
            agent_name="test",
            metadata=original_metadata,
            model_id="org/model",
            files=["shard.safetensors"],
            tp_rank=0,
        )
        packed = msgpack.packb(msg.model_dump(), use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        restored = ModelLoaded.model_validate(unpacked)
        assert restored.metadata == original_metadata

    def test_response_discriminated_union(self) -> None:
        """decode_response uses pydantic discriminated union dispatch."""
        for payload, expected_type in [
            (
                {
                    "type": "prepared",
                    "files": ["shard.safetensors"],
                    "peers": [],
                },
                Prepared,
            ),
            ({"type": "ok"}, Ok),
            ({"type": "error", "message": "boom"}, Error),
        ]:
            wire = msgpack.packb(payload, use_bin_type=True)
            result = decode_response(wire)
            assert isinstance(result, expected_type)

    def test_invalid_type_returns_raw_dict(self) -> None:
        """Unknown type fields fall through to raw dict."""
        wire = msgpack.packb(
            {"type": "unknown_future_type", "data": 123}, use_bin_type=True
        )
        result = decode_response(wire)
        assert isinstance(result, dict)
        assert result["type"] == "unknown_future_type"

    def test_peer_nixl_md_nested_validation(self) -> None:
        """PeerNixlMd nested inside Prepared validates correctly."""
        wire = msgpack.packb(
            {
                "type": "prepared",
                "files": ["shard.safetensors"],
                "peers": [
                    {"agent_name": "peer-0", "metadata": b"\x01\x02\x03"},
                    {"agent_name": "peer-1", "metadata": b"\x04\x05\x06"},
                ],
            },
            use_bin_type=True,
        )
        result = decode_response(wire)
        assert isinstance(result, Prepared)
        assert len(result.peers) == 2
        assert result.peers[0].agent_name == "peer-0"
        assert result.peers[1].metadata == b"\x04\x05\x06"


# Mock NIXL agent tests


@dataclass
class FakeTensor:
    """Minimal stand-in for torch.Tensor for testing."""

    _data_ptr: int
    _nelement: int
    _element_size: int
    _device_index: int

    def data_ptr(self) -> int:
        return self._data_ptr

    def nelement(self) -> int:
        return self._nelement

    def element_size(self) -> int:
        return self._element_size

    @property
    def device(self) -> "FakeDevice":
        return FakeDevice(self._device_index)


@dataclass
class FakeDevice:
    index: int | None


class TestWeightManifest:
    """Test the VramNixlAgent manifest tracking with mock tensors."""

    def test_manifest_populated_on_register(self) -> None:
        tensors = {
            "layer.0.weight": FakeTensor(0x7F0000, 1024, 4, 0),
            "layer.0.bias": FakeTensor(0x7F1000, 256, 4, 0),
        }

        manifest: list[dict] = []
        for name, t in tensors.items():
            manifest.append(
                {
                    "name": name,
                    "addr": t.data_ptr(),
                    "size": t.nelement() * t.element_size(),
                    "device_id": t.device.index if t.device.index is not None else 0,
                }
            )

        assert len(manifest) == 2
        by_name = {m["name"]: m for m in manifest}
        assert by_name["layer.0.weight"]["addr"] == 0x7F0000
        assert by_name["layer.0.weight"]["size"] == 4096
        assert by_name["layer.0.bias"]["size"] == 1024
        assert by_name["layer.0.bias"]["device_id"] == 0

    def test_manifest_survives_compound_roundtrip(self) -> None:
        tensors = {
            "attn.q": FakeTensor(0xAAAA0000, 2048, 2, 0),
            "attn.k": FakeTensor(0xBBBB0000, 2048, 2, 0),
            "attn.v": FakeTensor(0xCCCC0000, 2048, 2, 1),
        }

        manifest = [
            {
                "name": name,
                "addr": t.data_ptr(),
                "size": t.nelement() * t.element_size(),
                "device_id": t.device.index or 0,
            }
            for name, t in tensors.items()
        ]

        compound = msgpack.packb(
            {"nixl_md": b"\x00" * 64, "tensors": manifest},
            use_bin_type=True,
        )

        decoded = msgpack.unpackb(compound, raw=False)
        remote = {t["name"]: t for t in decoded["tensors"]}

        assert remote["attn.q"]["addr"] == 0xAAAA0000
        assert remote["attn.k"]["addr"] == 0xBBBB0000
        assert remote["attn.v"]["addr"] == 0xCCCC0000
        assert remote["attn.v"]["device_id"] == 1

    def test_manifest_with_checksums_roundtrip(self) -> None:
        """Checksum field survives msgpack round-trip."""
        tensors = [
            {
                "name": "layer.0.weight",
                "addr": 0x7F0000,
                "size": 4096,
                "device_id": 0,
                "dtype": "torch.bfloat16",
                "checksum": 0xDEADBEEFCAFEBABE,
            },
            {
                "name": "layer.0.bias",
                "addr": 0x7F1000,
                "size": 128,
                "device_id": 0,
                "dtype": "torch.float32",
                "checksum": 0x1234567890ABCDEF,
            },
        ]
        compound = msgpack.packb(
            {"nixl_md": b"\x00" * 64, "tensors": tensors},
            use_bin_type=True,
        )
        decoded = msgpack.unpackb(compound, raw=False)
        remote = {t["name"]: t for t in decoded["tensors"]}

        assert remote["layer.0.weight"]["checksum"] == 0xDEADBEEFCAFEBABE
        assert remote["layer.0.bias"]["checksum"] == 0x1234567890ABCDEF

    def test_manifest_without_checksum_backward_compat(self) -> None:
        """Old-style manifest (no checksum field) still works for consumers."""
        tensors = [
            {"name": "w", "addr": 100, "size": 4096, "device_id": 0},
        ]
        compound = msgpack.packb(
            {"nixl_md": b"\x00", "tensors": tensors},
            use_bin_type=True,
        )
        decoded = msgpack.unpackb(compound, raw=False)
        remote = {t["name"]: t for t in decoded["tensors"]}
        # Consumer uses .get("checksum") which returns None for missing key
        assert remote["w"].get("checksum") is None

    def test_manifest_with_none_checksum(self) -> None:
        """Checksum explicitly set to None (checksums disabled on source)."""
        tensors = [
            {
                "name": "w",
                "addr": 100,
                "size": 4096,
                "device_id": 0,
                "checksum": None,
            },
        ]
        compound = msgpack.packb(
            {"nixl_md": b"\x00", "tensors": tensors},
            use_bin_type=True,
        )
        decoded = msgpack.unpackb(compound, raw=False)
        remote = {t["name"]: t for t in decoded["tensors"]}
        assert remote["w"].get("checksum") is None

    def test_missing_tensor_detection(self) -> None:
        remote_tensors = [
            {"name": "layer.0.weight", "addr": 100, "size": 4096, "device_id": 0},
        ]
        remote_manifest = {t["name"]: t for t in remote_tensors}

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
