from __future__ import annotations

import torch
import intent
import intent.language as I


SHIFT_THRESHOLD = 8.0


@intent.fn
def _power_positive(base: I.f32, exponent: I.f32) -> I.f32:
    return I.exp(-exponent * I.log(base))


@intent.fn
def _tail_asymptotic(
    base: I.f32,
    exponent: I.f32,
    partial_sum: I.f32,
) -> I.f32:
    if base != base:
        return base
    infinite = I.cast(I.inf, I.f32)
    zero = I.cast(0.0, I.f32)
    if base == infinite:
        return zero

    inverse_power = _power_positive(base, exponent)
    if inverse_power != inverse_power:
        return partial_sum
    if inverse_power == infinite:
        return partial_sum
    if partial_sum != partial_sum:
        return partial_sum
    if partial_sum == infinite:
        return partial_sum

    inverse_base = I.cast(1.0, I.f32) / base
    inverse_base2 = inverse_base * inverse_base
    inverse_base3 = inverse_base2 * inverse_base
    inverse_base5 = inverse_base3 * inverse_base2
    inverse_base7 = inverse_base5 * inverse_base2
    inverse_base9 = inverse_base7 * inverse_base2

    one = I.cast(1.0, I.f32)
    factor1 = exponent
    factor3 = factor1 * (exponent + one) * (exponent + one + one)
    factor5 = factor3 * (exponent + one + one + one) * (
        exponent + one + one + one + one
    )
    factor7 = factor5 * (
        exponent + one + one + one + one + one
    ) * (
        exponent + one + one + one + one + one + one
    )
    factor9 = factor7 * (
        exponent + one + one + one + one + one + one + one
    ) * (
        exponent + one + one + one + one + one + one + one + one
    )

    tail = inverse_power * base / (exponent - I.cast(1.0, I.f32))
    tail = tail + I.cast(0.5, I.f32) * inverse_power
    tail = tail + inverse_power * inverse_base * factor1 / I.cast(
        12.0, I.f32
    )
    tail = tail - inverse_power * inverse_base3 * factor3 / I.cast(
        720.0, I.f32
    )
    tail = tail + inverse_power * inverse_base5 * factor5 / I.cast(
        30240.0, I.f32
    )
    tail = tail - inverse_power * inverse_base7 * factor7 / I.cast(
        1209600.0, I.f32
    )
    tail = tail + inverse_power * inverse_base9 * factor9 / I.cast(
        479001600.0, I.f32
    )
    return partial_sum + tail


@intent.kernel
def _zeta_shift(
    x_view: I.In[I.f32, ("N",)],
    q_view: I.In[I.f32, ("N",)],
    partial_sum: I.Out[I.f32, ("N",)],
    shifted_q: I.Out[I.f32, ("N",)],
    SHIFT_THRESHOLD: I.Constexpr[float],
):
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    infinite = I.cast(I.inf, I.f32)

    for i in I.parallel(I.domain(0, x_view.shape[0])):
        x_value = x_view[i]
        q_value = q_view[i]

        if x_value != x_value:
            partial_sum[i] = zero
            shifted_q[i] = q_value
        elif x_value < one:
            partial_sum[i] = zero
            shifted_q[i] = q_value
        elif x_value == one:
            partial_sum[i] = zero
            shifted_q[i] = q_value
        elif x_value == infinite:
            partial_sum[i] = zero
            shifted_q[i] = q_value
        elif q_value >= SHIFT_THRESHOLD:
            partial_sum[i] = zero
            shifted_q[i] = q_value
        elif q_value > zero:
            term = _power_positive(q_value, x_value)
            running_sum = zero
            shifted = q_value
            while shifted < SHIFT_THRESHOLD:
                running_sum = running_sum + term
                shifted = shifted + one
                term = term / shifted
            partial_sum[i] = running_sum
            shifted_q[i] = shifted
        else:
            partial_sum[i] = zero
            shifted_q[i] = q_value


@intent.kernel
def _zeta_finish(
    partial_sum: I.In[I.f32, ("N",)],
    shifted_q: I.In[I.f32, ("N",)],
    x_view: I.In[I.f32, ("N",)],
    out_view: I.Out[I.f32, ("N",)],
):
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    infinite = I.cast(I.inf, I.f32)

    for i in I.parallel(I.domain(0, out_view.shape[0])):
        x_value = x_view[i]
        q_value = shifted_q[i]

        if x_value != x_value:
            result = x_value
        elif x_value < one:
            result = zero / zero
        elif x_value == one:
            result = infinite
        elif x_value == infinite:
            result = zero
        elif q_value != q_value:
            result = q_value
        elif q_value > zero:
            result = _tail_asymptotic(
                q_value,
                x_value,
                partial_sum[i],
            )
        elif q_value == I.floor(q_value):
            result = infinite
        else:
            result = zero / zero

        out_view[i] = result


def build(context):
    shift_artifact = context.compile(
        "hurwitz_zeta_shift",
        _zeta_shift,
        constexprs={"SHIFT_THRESHOLD": SHIFT_THRESHOLD},
    )
    finish_artifact = context.compile("hurwitz_zeta_finish", _zeta_finish)
    workspace_by_invocation = {}

    def zeta(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None):
        element_count = input.shape[0]
        workspace_key = (element_count, input.device)
        workspace = workspace_by_invocation.get(workspace_key)
        if workspace is None:
            workspace = torch.empty(
                (2, element_count),
                dtype=torch.float32,
                device=input.device,
            )
            workspace_by_invocation[workspace_key] = workspace
        partial_sum = workspace[0]
        shifted_q = workspace[1]
        output = out if out is not None else torch.empty(
            (element_count,), dtype=torch.float32, device=input.device
        )

        shift_artifact(partial_sum, shifted_q, input, other)
        finish_artifact(partial_sum, shifted_q, input, output)
        return output

    return zeta
