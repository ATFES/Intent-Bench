from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _exp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.exp(values), mask=mask)


def build(context):
    def exp(input_tensor, out=None):
        output_tensor = torch.empty_like(input_tensor) if out is None else out
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _exp_kernel[grid](
            input_tensor,
            output_tensor,
            n_elements,
            BLOCK=1024,
            num_warps=4,
        )
        return output_tensor

    return exp
