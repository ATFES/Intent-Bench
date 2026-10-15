from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ifftshift_1d(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    half_size,
    BLOCK_SIZE: tl.constexpr,
):
    program_offset = tl.program_id(0) * BLOCK_SIZE
    output_offsets = program_offset + tl.arange(0, BLOCK_SIZE)
    valid = output_offsets < n_elements
    input_offsets = output_offsets + half_size
    input_offsets = tl.where(input_offsets >= n_elements, input_offsets - n_elements, input_offsets)
    values = tl.load(input_ptr + input_offsets * input_stride, mask=valid)
    tl.store(output_ptr + output_offsets, values, mask=valid)


def build(context):
    def ifftshift(input: torch.Tensor, dim=None) -> torch.Tensor:
        if input.ndim != 1:
            raise ValueError("this kernel supports the supplied one-dimensional profile")
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")

        if dim is None:
            selected_dim = 0
        elif isinstance(dim, int):
            selected_dim = dim
        elif isinstance(dim, tuple):
            if len(dim) != 1:
                raise ValueError("this kernel supports at most one dimension")
            selected_dim = dim[0]
        else:
            raise TypeError("dim must be None, an int, or a tuple of ints")

        if selected_dim < 0:
            selected_dim += input.ndim
        if selected_dim != 0:
            raise ValueError("dimension out of range")

        output = input.new_empty(input.shape)
        n_elements = input.numel()
        if n_elements == 0:
            return output

        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _ifftshift_1d[grid](
            input,
            output,
            n_elements,
            input.stride(0),
            n_elements // 2,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return output

    return ifftshift
