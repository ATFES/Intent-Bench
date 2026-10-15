"""Triton implementation of torch.max(input, dim, keepdim=False) for the
fixed 1-D float32 profile (N = 1,048,576, dim = 0, keepdim = False).

The reduction is split into two stages:
  1. every program reduces a contiguous BLOCK_SIZE chunk to a local
     (maximum, first-argmax-index) pair;
  2. a single program reduces those partial pairs to the final scalar pair.

The combine function matches PyTorch's float max/argmax semantics: NaN is
propagated, ties (including equal zeros of either sign) choose the smaller
index, and the selected value bit pattern is preserved.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _max_argmax_combine(v1, i1, v2, i2):
    """Combine two (value, index) groups, preserving PyTorch max semantics."""
    nan1 = v1 != v1
    nan2 = v2 != v2
    both_nan = nan1 & nan2

    normal = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    choose_left = tl.where(nan1 != nan2, nan1,
                           tl.where(both_nan, i1 < i2, normal))
    return tl.where(choose_left, v1, v2), tl.where(choose_left, i1, i2)


@triton.jit
def _partial_max_argmax_kernel(x_ptr, val_ptr, idx_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    values = tl.load(x_ptr + offsets)
    best_value, best_index = tl.reduce((values, offsets), 0, _max_argmax_combine)

    tl.store(val_ptr + pid, best_value)
    tl.store(idx_ptr + pid, best_index)


@triton.jit
def _final_max_argmax_kernel(val_ptr, idx_ptr, out_val_ptr, out_idx_ptr,
                             BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)

    values = tl.load(val_ptr + offsets)
    indices = tl.load(idx_ptr + offsets)
    best_value, best_index = tl.reduce((values, indices), 0, _max_argmax_combine)

    tl.store(out_val_ptr, best_value)
    tl.store(out_idx_ptr, best_index.to(tl.int64))


_BLOCK_SIZE = 4096
_NUM_WARPS = 8
_FINAL_NUM_WARPS = 4


def _next_power_of_two(n: int) -> int:
    return 1 << ((n - 1).bit_length())


def build(context):
    """Return a callable with signature max(input, dim, keepdim=False)."""

    def max_impl(input, dim, keepdim=False):
        # The supplied profile is a contiguous 1-D tensor and dim == 0.
        n = input.numel()
        if n == 0:
            raise ValueError("max(): expected reduction over a non-empty dimension")

        partials = triton.cdiv(n, _BLOCK_SIZE)
        final_block = _next_power_of_two(partials)

        partial_values = torch.empty(
            (partials,), dtype=torch.float32, device=input.device)
        partial_indices = torch.empty(
            (partials,), dtype=torch.int32, device=input.device)

        out_shape = (1,) if keepdim else ()
        out_values = torch.empty(out_shape, dtype=torch.float32, device=input.device)
        out_indices = torch.empty(out_shape, dtype=torch.int64, device=input.device)

        _partial_max_argmax_kernel[(partials,)](
            input, partial_values, partial_indices,
            BLOCK=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )

        _final_max_argmax_kernel[(1,)](
            partial_values, partial_indices, out_values, out_indices,
            BLOCK=final_block,
            num_warps=_FINAL_NUM_WARPS,
        )

        return torch.return_types.max((out_values, out_indices))

    return max_impl
