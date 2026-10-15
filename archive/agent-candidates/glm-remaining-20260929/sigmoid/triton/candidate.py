from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 2048


@triton.jit
def _sigmoid_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    stable_exponent = tl.exp(-tl.abs(input_value))
    positive_sigmoid = 1.0 / (1.0 + stable_exponent)
    negative_sigmoid = stable_exponent / (1.0 + stable_exponent)
    output_value = tl.where(input_value >= 0.0, positive_sigmoid, negative_sigmoid)
    tl.store(output_ptr + offsets, output_value, mask=mask)


def build(context):
    def sigmoid(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        grid = (triton.cdiv(input.numel(), BLOCK_SIZE),)
        _sigmoid_kernel[grid](
            input,
            output,
            input.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return output

    return sigmoid
