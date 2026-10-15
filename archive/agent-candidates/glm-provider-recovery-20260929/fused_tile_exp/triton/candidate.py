from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 1024


@triton.jit
def _fused_tile_exp_1d(
    input_ptr,
    output_ptr,
    input_numel: tl.constexpr,
    output_numel: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    source_offsets = offsets % input_numel
    values = tl.load(input_ptr + source_offsets)
    tl.store(output_ptr + offsets, tl.exp(values))


@triton.jit
def _fused_tile_exp_generic(
    input_ptr,
    output_ptr,
    output_numel,
    input_shape: tl.constexpr,
    input_strides: tl.constexpr,
    repeats: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < output_numel
    remaining = offsets.to(tl.int64)
    source_offset = tl.zeros_like(remaining)

    for axis in tl.static_range(len(repeats)):
        tiled_size = input_shape[axis] * repeats[axis]
        output_axis = remaining % tiled_size
        remaining //= tiled_size
        input_axis = output_axis // repeats[axis]
        source_offset += input_axis * input_strides[axis]

    values = tl.load(input_ptr + source_offset, mask=valid, other=0.0)
    tl.store(output_ptr + offsets, tl.exp(values), mask=valid)


def _normalized_tiling(shape: Sequence[int], dims: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rank = len(shape)
    dims = tuple(int(value) for value in dims)
    if len(dims) < rank:
        dims = (1,) * (rank - len(dims)) + dims
    elif len(dims) > rank:
        shape = (1,) * (len(dims) - rank) + tuple(shape)
    if any(value < 0 for value in dims):
        raise ValueError("tiling dimensions must be non-negative")
    return tuple(shape), dims


def build(context) -> Callable[[torch.Tensor, Sequence[int]], torch.Tensor]:
    del context

    def fused_tile_exp(
        input: torch.Tensor,
        dims: Sequence[int],
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape, repeats = _normalized_tiling(input.shape, dims)
        output_shape = tuple(shape_size * repeat for shape_size, repeat in zip(shape, repeats))
        output_numel = math.prod(output_shape)

        if out is None:
            output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        else:
            if out.shape != output_shape or out.dtype != input.dtype or out.device != input.device:
                raise ValueError("out must have the tiled shape, input dtype, and input device")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            output = out

        if output_numel == 0:
            return output

        if (
            input.ndim == 1
            and len(dims) == 1
            and input.is_contiguous()
            and input.is_cuda
            and output_numel % _BLOCK_SIZE == 0
            and output_numel <= 0x7FFFFFFF
        ):
            input_numel = input.numel()
            grid = (output_numel // _BLOCK_SIZE,)
            _fused_tile_exp_1d[grid](
                input,
                output,
                input_numel=input_numel,
                output_numel=output_numel,
                block_size=_BLOCK_SIZE,
            )
            return output

        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        if output_numel > 0x7FFFFFFF:
            raise ValueError("output tensor is too large for this kernel")

        if len(repeats) > input.ndim:
            effective_strides = (1,) * (len(repeats) - input.ndim) + tuple(input.stride())
        else:
            effective_strides = tuple(input.stride())
        grid = (triton.cdiv(output_numel, _BLOCK_SIZE),)
        _fused_tile_exp_generic[grid](
            input,
            output,
            output_numel,
            input_shape=shape,
            input_strides=effective_strides,
            repeats=repeats,
            block_size=_BLOCK_SIZE,
        )
        return output

    return fused_tile_exp
