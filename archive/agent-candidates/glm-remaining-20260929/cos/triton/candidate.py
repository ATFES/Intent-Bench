from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cos_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.cos(values), mask=mask)


def build(context):
    def cos(input_tensor, *, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)

        numel = input_tensor.numel()
        grid = (triton.cdiv(numel, 1024),)
        _cos_kernel[grid](input_tensor, out, numel, BLOCK_SIZE=1024)
        return out

    return cos
