from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _exp_sqrt_kernel(
    input_ptr,
    output_ptr,
    element_count,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < element_count
    values = tl.load(input_ptr + offsets, mask=valid)
    exponentials = tl.exp(values)
    tl.store(output_ptr + offsets, tl.sqrt_rn(exponentials), mask=valid)


def build(context):
    def exp_sqrt(input, out=None):
        output = torch.empty_like(input) if out is None else out
        element_count = input.numel()
        grid = (triton.cdiv(element_count, 1024),)
        _exp_sqrt_kernel[grid](
            input,
            output,
            element_count,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return exp_sqrt
