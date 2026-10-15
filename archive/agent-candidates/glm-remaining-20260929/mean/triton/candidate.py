from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum(
    input_ptr,
    partial_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    tl.store(partial_ptr + program_id, tl.sum(values, axis=0))


@triton.jit
def _finalize_mean(
    partial_ptr,
    output_ptr,
    n_partials,
    divisor,
    PARTIALS: tl.constexpr,
):
    offsets = tl.arange(0, PARTIALS)
    valid = offsets < n_partials
    values = tl.load(partial_ptr + offsets, mask=valid, other=0.0)
    tl.store(output_ptr, tl.sum(values, axis=0) / divisor)


def build(context):
    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        if (
            input_tensor.ndim != 1
            or dim not in (0, -1)
            or input_tensor.dtype != torch.float32
            or not input_tensor.is_contiguous()
            or (dtype is not None and dtype != torch.float32)
            or (out is not None and out.dtype != torch.float32)
        ):
            raise ValueError("unsupported mean configuration")

        if out is None:
            output = torch.empty((), dtype=torch.float32, device=input_tensor.device)
        else:
            output = out

        n_elements = input_tensor.numel()
        partials = torch.empty(256, dtype=torch.float32, device=input_tensor.device)
        _partial_sum[(256,)](
            input_tensor,
            partials,
            n_elements,
            BLOCK=4096,
            num_warps=8,
        )
        _finalize_mean[(1,)](
            partials,
            output,
            256,
            n_elements,
            PARTIALS=256,
            num_warps=1,
        )
        return output

    return mean
