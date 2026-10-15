from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        input_values = tl.load(input_ptr + offsets)
        other_values = tl.load(other_ptr + offsets)
        tl.store(output_ptr + offsets, input_values + alpha * other_values)
    else:
        valid = offsets < n_elements
        input_values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
        other_values = tl.load(other_ptr + offsets, mask=valid, other=0.0)
        tl.store(output_ptr + offsets, input_values + alpha * other_values, mask=valid)


def build(context):
    def add(input, other, alpha=1, out=None):
        if not (
            input.is_cuda
            and other.is_cuda
            and input.dtype is torch.float32
            and other.dtype is torch.float32
            and input.is_contiguous()
            and other.is_contiguous()
            and input.numel() == other.numel()
        ):
            raise NotImplementedError

        if out is None:
            output = torch.empty_like(input, memory_format=torch.contiguous_format)
        else:
            output = out
            if not output.is_contiguous():
                raise NotImplementedError

        count = input.numel()
        if count != 0:
            _add_kernel[(triton.cdiv(count, 4096),)](
                input,
                other,
                output,
                count,
                alpha,
                BLOCK_SIZE=4096,
                EVEN=count % 4096 == 0,
                num_warps=8,
            )
        return output

    return add
