from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _log1p_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.log1p(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def log1p(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _log1p_kernel[grid](input, result, n_elements, BLOCK=1024, num_warps=4)
        return result

    return log1p
