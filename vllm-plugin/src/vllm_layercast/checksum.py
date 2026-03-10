"""Per-tensor checksums for NIXL transfer integrity verification.

Uses a wyhash-inspired multiply-xor-shift hash. Triton kernel for GPU
tensors (hashes directly in VRAM, no D2H copy), pure Python fallback
for CPU tensors and testing.

Performance: ~100 GB/s on modern GPUs. A 140GB model hashes in ~1.4s,
negligible vs the 3-8s RDMA transfer time.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vllm_layercast.log import get_logger

log = get_logger("vllm_layercast.checksum")

if TYPE_CHECKING:
    import torch

# wyhash constants
_K1 = 0xA0761D6478BD642F
_K2 = 0xE7037ED1A0B428DB
_MASK = 0xFFFFFFFFFFFFFFFF

# Triton kernel (defined at module level, None if triton unavailable)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _xor_combine(a, b):
        return a ^ b

    @triton.jit
    def _wyhash_kernel(
        data_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint64)
        indices = offsets.to(tl.uint64)

        # wyhash-style multiply-xor-shift mixing
        K1: tl.constexpr = 0xA0761D6478BD642F
        K2: tl.constexpr = 0xE7037ED1A0B428DB

        mixed = data ^ (indices * K2)
        mixed = mixed * K1
        mixed = mixed ^ (mixed >> 32)

        zero = tl.zeros([BLOCK_SIZE], dtype=tl.uint64)
        mixed = tl.where(mask, mixed, zero)

        partial = tl.reduce(mixed, axis=0, combine_fn=_xor_combine)
        tl.atomic_xor(output_ptr, partial.to(tl.int64))

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def is_enabled() -> bool:
    """Check if checksum verification is enabled via LAYERCAST_CHECKSUM env var."""
    return os.environ.get("LAYERCAST_CHECKSUM", "true").lower() not in (
        "false",
        "0",
        "no",
    )


def wyhash_cpu(data: bytes) -> int:
    """Pure Python wyhash-style hash over raw bytes.

    Processes data as little-endian uint64 chunks with position-dependent
    mixing. Remainder bytes are zero-padded to 8 bytes.
    """
    if not data:
        return 0

    n_u64 = len(data) // 8
    remainder = len(data) % 8
    h = 0

    for i in range(n_u64):
        v = int.from_bytes(data[i * 8 : (i + 1) * 8], "little")
        mixed = ((v ^ ((i * _K2) & _MASK)) * _K1) & _MASK
        mixed ^= mixed >> 32
        h ^= mixed

    if remainder:
        last_bytes = data[n_u64 * 8 :] + b"\x00" * (8 - remainder)
        last = int.from_bytes(last_bytes, "little")
        mixed = ((last ^ ((n_u64 * _K2) & _MASK)) * _K1) & _MASK
        mixed ^= mixed >> 32
        h ^= mixed

    return h


def _wyhash_triton(tensor: "torch.Tensor") -> int:
    """Compute wyhash of a CUDA tensor using the Triton kernel."""
    import torch

    flat = tensor.contiguous().view(torch.uint8)
    n_bytes = flat.numel()

    if n_bytes == 0:
        return 0

    # Pad to multiple of 8 bytes so we can reinterpret as int64
    remainder = n_bytes % 8
    if remainder:
        flat = torch.nn.functional.pad(flat, (0, 8 - remainder))

    data_i64 = flat.view(torch.int64)
    n_elements = data_i64.numel()

    output = torch.zeros(1, dtype=torch.int64, device=tensor.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _wyhash_kernel[grid](data_i64, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Convert signed int64 to unsigned Python int
    return output.item() & _MASK


def tensor_wyhash(tensor: "torch.Tensor") -> int:
    """Compute wyhash checksum of a tensor's raw bytes.

    Uses Triton kernel on CUDA tensors for zero-copy hashing directly
    in VRAM. Falls back to pure Python for CPU tensors.
    """
    import torch

    if tensor.numel() == 0:
        return 0

    if tensor.is_cuda and _HAS_TRITON:
        return _wyhash_triton(tensor)

    # CPU fallback
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return wyhash_cpu(raw)


def compute_checksums(tensors: dict[str, "torch.Tensor"]) -> dict[str, int]:
    """Batch wrapper: compute wyhash checksum for each tensor."""
    return {name: tensor_wyhash(t) for name, t in tensors.items()}


def verify_checksums(
    tensors: dict[str, "torch.Tensor"],
    expected: dict[str, int],
) -> list[str]:
    """Verify tensor checksums against expected values.

    Returns list of tensor names with mismatched checksums.
    Skips tensors not present in expected (backward compat with
    old sources that don't publish checksums).
    """
    mismatched: list[str] = []
    for name, tensor in tensors.items():
        if name not in expected:
            continue
        actual = tensor_wyhash(tensor)
        if actual != expected[name]:
            log.error(
                "checksum_mismatch",
                tensor=name,
                expected=hex(expected[name]),
                actual=hex(actual),
            )
            mismatched.append(name)
    return mismatched
