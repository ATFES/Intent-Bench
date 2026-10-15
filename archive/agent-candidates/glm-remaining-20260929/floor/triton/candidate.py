from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _floor_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=valid)
    tl.store(output_ptr + offsets, tl.floor(values), mask=valid)


@triton.jit
def _copy_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=valid)
    tl.store(output_ptr + offsets, values, mask=valid)


def build(context):
    def floor(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        count = input.numel()
        output = torch.empty_like(input) if out is None else out
        if input.is_floating_point():
            _floor_kernel[(triton.cdiv(count, 1024),)](
                input,
                output,
                count,
                BLOCK=1024,
                num_warps=4,
            )
        else:
            _copy_kernel[(triton.cdiv(count, 1024),)](
                input,
                output,
                count,
                BLOCK=1024,
                num_warps=4,
            )
        return output

    return floor
