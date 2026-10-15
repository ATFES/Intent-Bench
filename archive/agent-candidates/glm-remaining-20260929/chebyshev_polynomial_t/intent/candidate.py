from __future__ import annotations

from typing import Optional

import intent
import intent.language as I
import torch


@intent.kernel
def _chebyshev_recurrence(
    input: I.In[I.f32, ("N",)],
    degree: I.i64,
    out: I.Out[I.f32, ("N",)],
) -> None:
    domain = I.domain(0, input.shape[0])
    values = input[domain]
    previous = I.full(out.shape, 1.0, I.f32)
    current = values
    step = I.cast(0, I.i64)

    while step < degree:
        updated = 2.0 * values * current - previous
        previous = current
        current = updated
        step = step + 1

    result = previous
    if degree != 0:
        result = current

    if degree >= 6:
        magnitude = I.abs(values)
        inside = magnitude <= 1.0
        clamped = I.select(inside, values, I.cast(1.0, I.f32))
        low = I.cast(0.0, I.f32)
        high = I.cast(3.141592653589793, I.f32)
        for _ in range(48):
            middle = (low + high) * 0.5
            below = clamped <= I.cos(middle)
            low = I.select(below, middle, low)
            high = I.select(below, high, middle)
        angle = (low + high) * 0.5
        trigonometric = I.cos(I.cast(degree, I.f32) * angle)
        result = I.select(inside, trigonometric, result)

    out[domain] = result


def build(context):
    recurrence = context.compile(
        "chebyshev_polynomial_t_recurrence", _chebyshev_recurrence
    )

    def chebyshev_polynomial_t(
        input: torch.Tensor,
        n: int = 3,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        destination = out if out is not None else torch.empty_like(input)
        recurrence(input, n, destination)
        return destination

    return chebyshev_polynomial_t
