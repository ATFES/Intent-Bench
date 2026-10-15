from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.cuda import libdevice


BLOCK_SIZE = 1024


@triton.jit
def _chebyshev_polynomial_t_kernel(
    input_ptr,
    output_ptr,
    element_count,
    degree,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    value = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if degree == 0:
        result = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    elif degree == 1:
        result = value
    else:
        previous = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
        current = value
        recurrence_index = 2

        while recurrence_index <= degree:
            next_value = 2.0 * value * current - previous
            previous = current
            current = next_value
            recurrence_index += 1

        if degree < 6:
            result = current
        else:
            angle = libdevice.acos(value)
            trigonometric = tl.cos(degree.to(tl.float32) * angle)
            use_trigonometric = tl.abs(value) <= 1.0
            result = tl.where(use_trigonometric, trigonometric, current)

    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def chebyshev_polynomial_t(input: torch.Tensor, n: int, *, out=None):
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out

        element_count = input.numel()
        grid = (triton.cdiv(element_count, BLOCK_SIZE),)
        _chebyshev_polynomial_t_kernel[grid](
            input,
            output,
            element_count,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    return chebyshev_polynomial_t
