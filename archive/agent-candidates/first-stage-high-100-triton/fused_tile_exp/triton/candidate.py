from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tile_exp_kernel(x_ptr, out_ptr, total_elements, input_elements, BLOCK: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    # The fixed tiling is two repetitions of the one-dimensional input.
    source_offsets = tl.where(offsets < input_elements, offsets, offsets - input_elements)
    values = tl.load(x_ptr + source_offsets, mask=mask, other=0.0)
    result = tl.exp(values)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, dims, *, out=None):
        input_elements = input.numel()
        if out is None:
            out = torch.empty((2, input_elements), device=input.device, dtype=input.dtype)

        total_elements = 2 * input_elements
        grid = (triton.cdiv(total_elements, 1024),)
        _tile_exp_kernel[grid](
            input,
            out,
            total_elements,
            input_elements,
            BLOCK=1024,
            num_warps=8,
        )
        return out

    return wrapper
