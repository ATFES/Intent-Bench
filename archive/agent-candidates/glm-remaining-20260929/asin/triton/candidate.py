from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _asin_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    output_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_offset = tl.program_id(0) * BLOCK_SIZE
    offsets = program_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_offsets = offsets * input_stride
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    result = tl.extra.cuda.libdevice.asin(values.to(tl.float32))
    output_offsets = offsets * output_stride
    tl.store(output_ptr + output_offsets, result.to(tl.float16), mask=mask)


def build(context):
    def asin(input_tensor, *, out=None):
        output = torch.empty_like(input_tensor) if out is None else out
        n_elements = input_tensor.numel()
        block_size = 2048
        grid = (triton.cdiv(n_elements, block_size),)
        _asin_kernel[grid](
            input_tensor,
            output,
            input_tensor.stride(0),
            output.stride(0),
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        return output

    return asin
