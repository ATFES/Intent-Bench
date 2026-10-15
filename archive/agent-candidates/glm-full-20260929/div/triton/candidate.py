from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _div_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_SIZE:
        numerator = tl.load(input_ptr + offsets)
        denominator = tl.load(other_ptr + offsets)
        result = tl.math.div_rn(numerator, denominator)
        tl.store(output_ptr + offsets, result)
    else:
        mask = offsets < num_elements
        numerator = tl.load(input_ptr + offsets, mask=mask)
        denominator = tl.load(other_ptr + offsets, mask=mask)
        result = tl.math.div_rn(numerator, denominator)
        tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def div(input, other, rounding_mode=None, out=None):
        if rounding_mode is not None:
            raise ValueError("rounding_mode must be None for this profile")
        if input.dtype != torch.float32 or other.dtype != torch.float32:
            raise TypeError("this profile requires float32 operands")
        if input.device != other.device or input.device.type != "cuda":
            raise TypeError("operands must be CUDA tensors on the same device")
        if input.shape != other.shape or input.dim() != 1:
            raise ValueError("this profile requires equal one-dimensional shapes")
        if input.stride(0) != 1 or other.stride(0) != 1:
            raise ValueError("this profile requires contiguous operands")

        num_elements = input.numel()
        if out is None:
            output = torch.empty_like(input)
        else:
            if out.dtype != torch.float32 or out.shape != input.shape:
                raise ValueError("out must have the profile's shape and dtype")
            if out.device != input.device or out.stride(0) != 1:
                raise ValueError("out must be contiguous and on the input device")
            output = out

        block_size = 1024
        even_size = num_elements % block_size == 0
        grid = (triton.cdiv(num_elements, block_size),)
        _div_kernel[grid](
            input,
            other,
            output,
            num_elements,
            BLOCK_SIZE=block_size,
            EVEN_SIZE=even_size,
            num_warps=4,
        )
        return output

    return div
