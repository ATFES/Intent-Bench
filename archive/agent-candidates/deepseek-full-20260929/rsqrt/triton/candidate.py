from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rsqrt_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    if EXACT:
        x = tl.load(x_ptr + offsets)
        y = tl.rsqrt(x)
        tl.store(y_ptr + offsets, y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.rsqrt(x)
        tl.store(y_ptr + offsets, y, mask=mask)


def build(context):
    def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)

        n_elements = input.numel()
        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        _rsqrt_kernel[grid](
            input,
            out,
            n_elements,
            BLOCK=block,
            EXACT=(n_elements % block == 0),
            num_warps=8,
        )
        return out

    return rsqrt
