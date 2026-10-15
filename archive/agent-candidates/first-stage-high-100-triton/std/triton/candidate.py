from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_welford(x_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    count = tl.sum(mask.to(tl.float32), axis=0)
    total = tl.sum(values, axis=0)
    mean = total / count
    centered = values - mean
    m2 = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0)

    base = pid * 3
    tl.store(partial_ptr + base, mean)
    tl.store(partial_ptr + base + 1, m2)
    tl.store(partial_ptr + base + 2, count)


@triton.jit
def _finish_std(partial_ptr, output_ptr, correction, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    means = tl.load(partial_ptr + offsets * 3)
    m2 = tl.load(partial_ptr + offsets * 3 + 1)
    counts = tl.load(partial_ptr + offsets * 3 + 2)

    total_count = tl.sum(counts, axis=0)
    mean = tl.sum(means * counts, axis=0) / total_count
    between = tl.sum(counts * (means - mean) * (means - mean), axis=0)
    variance = (tl.sum(m2, axis=0) + between) / tl.maximum(total_count - correction, 0.0)
    tl.store(output_ptr, tl.sqrt(variance))


def build(context):
    def wrapper(input, dim=None, *, correction=1, keepdim=False, out=None):
        n_elements = input.numel()
        block = 1024
        num_programs = triton.cdiv(n_elements, block)
        partials = torch.empty((num_programs, 3), device=input.device, dtype=torch.float32)
        result = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)

        _partial_welford[(num_programs,)](
            input,
            partials,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )
        _finish_std[(1,)](
            partials,
            result,
            correction,
            BLOCK=num_programs,
            num_warps=8,
        )
        return result

    return wrapper
