from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    numel,
    scalar_other,
    OTHER_IS_SCALAR: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if OTHER_IS_SCALAR:
        other_values = scalar_other
    else:
        other_values = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    products = input_values * other_values
    results = tl.maximum(products, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(output_ptr + offsets, results, mask=mask)


def build(context):
    del context

    def mul_relu(input, other, inplace=False, out=None):
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if input.dtype != torch.float32:
            raise TypeError("input must be float32")
        if input.device.type != "cuda":
            raise TypeError("input must be on a CUDA device")

        if isinstance(other, torch.Tensor):
            other_is_scalar = False
            scalar_other = 0.0
            if other.dtype != torch.float32:
                raise TypeError("other must be float32")
            if other.device != input.device:
                raise TypeError("input and other must be on the same device")
            if other.shape != input.shape:
                raise ValueError("input and other must have the same shape")
            if not other.is_contiguous():
                raise ValueError("other must be contiguous")
        elif isinstance(other, (int, float)):
            other_is_scalar = True
            scalar_other = float(other)
        else:
            raise TypeError("other must be a tensor or number")

        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if inplace and out is not None and out.data_ptr() != input.data_ptr():
            raise ValueError("out must alias input when inplace is true")
        if out is None:
            output = input if inplace else torch.empty_like(input)
        else:
            if not isinstance(out, torch.Tensor):
                raise TypeError("out must be a torch.Tensor")
            if out.dtype != torch.float32:
                raise TypeError("out must be float32")
            if out.device != input.device:
                raise TypeError("out must be on the same device as input")
            if out.shape != input.shape:
                raise ValueError("out must have the same shape as input")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            output = out

        numel = input.numel()
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _mul_relu_kernel[grid](
            input,
            None if other_is_scalar else other,
            output,
            numel,
            scalar_other,
            OTHER_IS_SCALAR=other_is_scalar,
        )
        return output

    return mul_relu
