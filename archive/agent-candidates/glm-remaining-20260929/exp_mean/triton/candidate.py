from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _exp_partial_mean_kernel(
    input_ptr,
    partials_ptr,
    numel,
    input_stride,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    tl.store(partials_ptr + program_id, tl.sum(tl.exp(values), axis=0))


@triton.jit
def _finalize_mean_kernel(
    partials_ptr,
    output_ptr,
    numel,
    num_partials,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_partials
    partials = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr, tl.sum(partials, axis=0) / numel)


def build(context):
    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        if dim is not None:
            raise ValueError("this candidate supports the fixed all-elements profile")
        if input.dtype != torch.float32 or input.device.type != "cuda":
            raise ValueError("input must be a CUDA float32 tensor")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")

        numel = input.numel()
        block_size = 1024
        num_partials = triton.cdiv(numel, block_size)
        final_block_size = triton.next_power_of_2(num_partials)

        output_dtype = torch.float32 if dtype is None else dtype
        if output_dtype != torch.float32:
            raise ValueError("output dtype must be float32")
        if out is None:
            output = torch.empty((), dtype=torch.float32, device=input.device)
        else:
            output = out

        partials = torch.empty(
            (num_partials,), dtype=torch.float32, device=input.device
        )
        _exp_partial_mean_kernel[(num_partials,)](
            input,
            partials,
            numel,
            input.stride(0),
            BLOCK=block_size,
            num_warps=4,
        )
        _finalize_mean_kernel[(1,)](
            partials,
            output,
            numel,
            num_partials,
            BLOCK=final_block_size,
            num_warps=8,
        )
        return output

    return exp_mean
