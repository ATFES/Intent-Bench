from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _erfc_sqrt_kernel(
    input_ptr,
    erfc_ptr,
    sqrt_ptr,
    n_elements,
    input_stride,
    erfc_stride,
    sqrt_stride,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    program_id = tl.program_id(0)
    if EVEN:
        offsets = program_id * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(input_ptr + offsets * input_stride)
        tl.store(erfc_ptr + offsets * erfc_stride, libdevice.erfc(values))
        tl.store(sqrt_ptr + offsets * sqrt_stride, tl.sqrt_rn(values))
    else:
        offsets = program_id * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < n_elements
        values = tl.load(input_ptr + offsets * input_stride, mask=valid, other=0.0)
        tl.store(erfc_ptr + offsets * erfc_stride, libdevice.erfc(values), mask=valid)
        tl.store(sqrt_ptr + offsets * sqrt_stride, tl.sqrt_rn(values), mask=valid)


def build(context):
    def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_elements = input.numel()
        erfc_output = torch.empty_like(input)
        sqrt_output = torch.empty_like(input)
        block_size = 1024
        even = n_elements % block_size == 0
        grid = (triton.cdiv(n_elements, block_size),)
        _erfc_sqrt_kernel[grid](
            input,
            erfc_output,
            sqrt_output,
            n_elements,
            input.stride(0),
            erfc_output.stride(0),
            sqrt_output.stride(0),
            BLOCK=block_size,
            EVEN=even,
            num_warps=4,
        )
        return erfc_output, sqrt_output

    return erfc_sqrt
