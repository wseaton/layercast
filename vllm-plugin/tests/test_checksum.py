"""Tests for per-tensor wyhash checksums.

Pure Python tests (wyhash_cpu, config) run everywhere.
Tensor tests require torch (skipped if unavailable).
GPU tests require CUDA + Triton (skipped if unavailable).
"""

from __future__ import annotations

import os
import struct

import pytest

from vllm_layercast.checksum import (
    _K1,
    _K2,
    _MASK,
    is_enabled,
    wyhash_cpu,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


# wyhash_cpu correctness (no torch needed)


class TestWyhashCpu:
    """Verify the pure Python wyhash reference implementation."""

    def test_empty_returns_zero(self) -> None:
        assert wyhash_cpu(b"") == 0

    def test_deterministic(self) -> None:
        data = b"the quick brown fox jumps over the lazy dog"
        h1 = wyhash_cpu(data)
        h2 = wyhash_cpu(data)
        assert h1 == h2

    def test_different_data_different_hash(self) -> None:
        h1 = wyhash_cpu(b"\x00" * 64)
        h2 = wyhash_cpu(b"\xff" * 64)
        assert h1 != h2

    def test_single_bit_flip_detected(self) -> None:
        """A single bit flip in the data must change the hash."""
        data = bytearray(b"\x00" * 128)
        h_original = wyhash_cpu(bytes(data))

        # Flip bit 0 of byte 0
        data[0] ^= 0x01
        h_flipped = wyhash_cpu(bytes(data))
        assert h_original != h_flipped

    def test_single_bit_flip_every_byte(self) -> None:
        """Flip one bit in every byte position, each must change the hash."""
        data = os.urandom(256)
        h_original = wyhash_cpu(data)

        for i in range(len(data)):
            corrupted = bytearray(data)
            corrupted[i] ^= 0x80  # flip high bit
            assert wyhash_cpu(bytes(corrupted)) != h_original, f"byte {i}"

    def test_remainder_handling(self) -> None:
        """Data sizes that aren't multiples of 8 must still hash correctly."""
        for size in [1, 3, 5, 7, 9, 15, 17, 63]:
            data = os.urandom(size)
            h = wyhash_cpu(data)
            assert h == wyhash_cpu(data)
            assert 0 <= h <= _MASK

    def test_exactly_8_bytes(self) -> None:
        """Single uint64 chunk, no remainder."""
        data = struct.pack("<Q", 0xDEADBEEFCAFEBABE)
        h = wyhash_cpu(data)
        assert 0 < h <= _MASK

    def test_position_sensitivity(self) -> None:
        """Swapping two 8-byte chunks must change the hash (position-dependent mixing)."""
        chunk_a = b"\x01" * 8
        chunk_b = b"\x02" * 8
        h1 = wyhash_cpu(chunk_a + chunk_b)
        h2 = wyhash_cpu(chunk_b + chunk_a)
        assert h1 != h2

    def test_known_value(self) -> None:
        """Pin a known hash to catch accidental algorithm changes."""
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        h = wyhash_cpu(data)
        # Compute expected manually
        v = int.from_bytes(data, "little")
        expected = ((v ^ ((0 * _K2) & _MASK)) * _K1) & _MASK
        expected ^= expected >> 32
        assert h == expected


# tensor_wyhash (CPU path, requires torch)


@requires_torch
class TestTensorWyhash:
    """Verify tensor_wyhash using CPU tensors (no GPU needed)."""

    def test_empty_tensor(self) -> None:
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.empty(0, dtype=torch.float32)
        assert tensor_wyhash(t) == 0

    def test_matches_cpu_reference(self) -> None:
        """tensor_wyhash on CPU must match wyhash_cpu on same bytes."""
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.randn(1024, dtype=torch.float32)
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        assert tensor_wyhash(t) == wyhash_cpu(raw)

    def test_matches_cpu_reference_bfloat16(self) -> None:
        """bfloat16 tensors must hash correctly (2 bytes per element)."""
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.randn(512, dtype=torch.bfloat16)
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        assert tensor_wyhash(t) == wyhash_cpu(raw)

    def test_matches_cpu_reference_int8(self) -> None:
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.randint(-128, 127, (333,), dtype=torch.int8)
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        assert tensor_wyhash(t) == wyhash_cpu(raw)

    def test_deterministic(self) -> None:
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.randn(2048, dtype=torch.float32)
        assert tensor_wyhash(t) == tensor_wyhash(t)

    def test_detects_corruption(self) -> None:
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.zeros(256, dtype=torch.float32)
        h_clean = tensor_wyhash(t)
        t[0] = 1.0
        h_corrupted = tensor_wyhash(t)
        assert h_clean != h_corrupted

    def test_multidimensional(self) -> None:
        """Shape shouldn't matter, only the raw bytes (contiguous layout)."""
        from vllm_layercast.checksum import tensor_wyhash

        t1d = torch.arange(24, dtype=torch.float32)
        t2d = t1d.reshape(4, 6)
        t3d = t1d.reshape(2, 3, 4)
        h1 = tensor_wyhash(t1d)
        assert tensor_wyhash(t2d) == h1
        assert tensor_wyhash(t3d) == h1

    def test_non_contiguous_handled(self) -> None:
        """Non-contiguous tensors must be made contiguous before hashing."""
        from vllm_layercast.checksum import tensor_wyhash

        t = torch.randn(10, 10, dtype=torch.float32)
        col = t[:, 0]  # non-contiguous slice
        assert not col.is_contiguous()
        h = tensor_wyhash(col)
        h_ref = tensor_wyhash(col.contiguous())
        assert h == h_ref


# compute_checksums / verify_checksums (requires torch)


@requires_torch
class TestBatchOperations:
    """Test the batch compute/verify wrappers."""

    def test_compute_checksums(self) -> None:
        from vllm_layercast.checksum import compute_checksums

        tensors = {
            "layer.0.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer.0.bias": torch.randn(64, dtype=torch.float32),
            "layer.1.weight": torch.randn(64, 64, dtype=torch.float32),
        }
        checksums = compute_checksums(tensors)
        assert len(checksums) == 3
        assert all(isinstance(v, int) for v in checksums.values())
        assert all(0 <= v <= _MASK for v in checksums.values())

    def test_verify_all_match(self) -> None:
        from vllm_layercast.checksum import compute_checksums, verify_checksums

        tensors = {
            "w1": torch.randn(100, dtype=torch.float32),
            "w2": torch.randn(200, dtype=torch.float32),
        }
        expected = compute_checksums(tensors)
        mismatched = verify_checksums(tensors, expected)
        assert mismatched == []

    def test_verify_detects_mismatch(self) -> None:
        from vllm_layercast.checksum import compute_checksums, verify_checksums

        tensors = {
            "w1": torch.randn(100, dtype=torch.float32),
            "w2": torch.randn(200, dtype=torch.float32),
        }
        expected = compute_checksums(tensors)
        # Corrupt w1
        tensors["w1"][0] = 999.0
        mismatched = verify_checksums(tensors, expected)
        assert "w1" in mismatched
        assert "w2" not in mismatched

    def test_verify_skips_missing_expected(self) -> None:
        """Tensors not in expected dict are silently skipped (backward compat)."""
        from vllm_layercast.checksum import tensor_wyhash, verify_checksums

        tensors = {
            "w1": torch.randn(100, dtype=torch.float32),
            "w2": torch.randn(200, dtype=torch.float32),
        }
        # Only provide checksum for w1
        expected = {"w1": tensor_wyhash(tensors["w1"])}
        mismatched = verify_checksums(tensors, expected)
        assert mismatched == []

    def test_verify_empty_expected(self) -> None:
        """No expected checksums means nothing to verify (old source)."""
        from vllm_layercast.checksum import verify_checksums

        tensors = {"w1": torch.randn(100, dtype=torch.float32)}
        mismatched = verify_checksums(tensors, {})
        assert mismatched == []


# Config toggle (no torch needed)


class TestConfig:
    """Verify the LAYERCAST_CHECKSUM env var toggle."""

    def test_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LAYERCAST_CHECKSUM", raising=False)
        assert is_enabled() is True

    def test_enabled_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYERCAST_CHECKSUM", "true")
        assert is_enabled() is True

    def test_disabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYERCAST_CHECKSUM", "false")
        assert is_enabled() is False

    def test_disabled_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYERCAST_CHECKSUM", "0")
        assert is_enabled() is False

    def test_disabled_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYERCAST_CHECKSUM", "no")
        assert is_enabled() is False

    def test_disabled_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LAYERCAST_CHECKSUM", "FALSE")
        assert is_enabled() is False
