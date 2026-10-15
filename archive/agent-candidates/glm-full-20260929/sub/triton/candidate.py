from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sub_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    numel,
    alpha,
    ALPHA_IS_ONE: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_SIZE:
        input_value = tl.load(input_ptr + offsets)
        other_value = tl.load(other_ptr + offsets)
        if ALPHA_IS_ONE:
            result = input_value - other_value
        else:
            result = input_value - alpha * other_value
        tl.store(output_ptr + offsets, result)
    else:
        mask = offsets < numel
        input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        other_value = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        if ALPHA_IS_ONE:
            result = input_value - other_value
        else:
            result = input_value - alpha * other_value
        tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def sub(input, other, *, alpha: float = 1, out=None):
        if out is None:
            out = torch.empty_like(input)

        numel = input.numel()
        block_size = 1024
        grid = (triton.cdiv(numel, block_size),)
        _sub_kernel[grid](
            input,
            other,
            out,
            numel,
            alpha,
            ALPHA_IS_ONE=(alpha == 1),
            EVEN_SIZE=(numel % block_size == 0),
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return out

    return sub
