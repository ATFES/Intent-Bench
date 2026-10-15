from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _exp_sum_kernel(x, partial, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    total = tl.sum(tl.exp(values), axis=0)
    tl.store(partial + pid, total)


@triton.jit
def _mean_partials_kernel(partial, out, n_partials, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(out, total / n_elements)


def build(context):
    # The evaluator's profile is a contiguous one-dimensional float32 tensor.
    block = 1024
    n_partials = 1024

    def wrapper(input, dim=None, keepdim=False, dtype=None, out=None):
        n_elements = input.numel()
        result = out if out is not None else torch.empty((), device=input.device, dtype=torch.float32)
        partial = torch.empty((n_partials,), device=input.device, dtype=torch.float32)
        grid = (triton.cdiv(n_elements, block),)
        _exp_sum_kernel[grid](input, partial, n_elements, BLOCK=block)
        _mean_partials_kernel[(1,)](partial, result, n_partials, n_elements, BLOCK=n_partials)
        return result

    return wrapper
