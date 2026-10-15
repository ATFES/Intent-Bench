from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _zeta_kernel(
    x_ptr,
    q_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=valid, other=1.0)
    q = tl.load(q_ptr + offsets, mask=valid, other=1.0)

    positive_x = x > 1.0
    nonpositive_q = q <= 0.0
    q_is_integer = q == tl.floor(q)

    finite_positive_q = (q > 0.0) & (q < float("inf"))
    base = q
    series = tl.zeros_like(q)

    for k in range(16):
        series += libdevice.pow(base, -x)
        base += 1.0

    z = base
    inverse_power = libdevice.pow(z, -x)
    half_term = 0.5 * inverse_power
    integral_term = libdevice.pow(z, 1.0 - x) / (x - 1.0)
    derivative_power = inverse_power
    derivative_term = -(x / 12.0) * derivative_power * (z * z)

    correction_power = inverse_power
    correction_product = x
    correction = (x / 12.0) * correction_power * (z * z)
    correction_power *= z * z
    correction_product *= (x + 1.0) * (x + 2.0)
    correction -= correction_product * correction_power / 120.0

    correction_power *= z * z
    correction_product *= (x + 3.0) * (x + 4.0)
    correction += correction_product * correction_power / 252.0

    correction_power *= z * z
    correction_product *= (x + 5.0) * (x + 6.0)
    correction -= correction_product * correction_power / 240.0

    correction_power *= z * z
    correction_product *= (x + 7.0) * (x + 8.0)
    correction += correction_product * correction_power / 132.0

    correction_power *= z * z
    correction_product *= (x + 9.0) * (x + 10.0)
    correction -= correction_product * correction_power * 691.0 / 32760.0

    use_long_correction = x <= 24.0
    tail = half_term + integral_term
    tail -= tl.where(use_long_correction, correction, derivative_term)
    positive_q_result = series + tail

    x_is_integer = x == tl.floor(x)
    half_x = 0.5 * x
    x_is_even = half_x == tl.floor(half_x)
    noninteger_q_integer_x_result = tl.where(x_is_even, float("inf"), float("-inf"))
    nonpositive_q_result = tl.where(
        q_is_integer,
        float("inf"),
        tl.where(x_is_integer, noninteger_q_integer_x_result, float("nan")),
    )

    result = tl.where(
        positive_x,
        tl.where(nonpositive_q, nonpositive_q_result, positive_q_result),
        tl.where(x == 1.0, float("inf"), float("nan")),
    )
    result = tl.where(finite_positive_q | positive_x, result, float("nan"))
    result = tl.where(x == float("inf"), 0.0, result)
    tl.store(out_ptr + offsets, result, mask=valid)


def build(context):
    def zeta(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None):
        output = out if out is not None else torch.empty_like(input)
        count = input.numel()
        _zeta_kernel[(triton.cdiv(count, 512),)](
            input,
            other,
            output,
            count,
            BLOCK=512,
        )
        return output

    return zeta
