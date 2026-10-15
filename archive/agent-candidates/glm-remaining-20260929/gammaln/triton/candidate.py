from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gammaln_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    results = libdevice.lgamma(values)
    tl.store(output_ptr + offsets, results, mask=mask)


def build(context):
    def gammaln(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        output = out if out is not None else torch.empty_like(input)
        numel = input.numel()
        block_size = 1024
        grid = ((numel + block_size - 1) // block_size,)
        _gammaln_kernel[grid](input, output, numel, BLOCK_SIZE=block_size)
        return output

    return gammaln
