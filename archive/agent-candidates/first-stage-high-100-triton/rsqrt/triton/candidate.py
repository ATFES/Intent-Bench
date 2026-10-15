from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.rsqrt(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _rsqrt_kernel[grid](input, out, n_elements, BLOCK=1024, num_warps=4)
        return out

    return wrapper
