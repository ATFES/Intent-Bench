from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _asin_elementwise(
    values: I.In[I.f16, ("N",)],
    result: I.Out[I.f16, ("N",)],
) -> None:
    for element in I.parallel(I.domain(0, values.shape[0])):
        value = I.cast(values[element], I.f32)
        magnitude = I.abs(value)

        low_y = value * value
        low_polynomial = I.cast(0.001373291015625, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.009761609511, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.011551800890, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.013964843750, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.017352764423, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.022372159090, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.030381944444, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.044642857143, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.075000000000, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(0.166666666667, I.f32)
        low_polynomial = low_polynomial * low_y + I.cast(1.000000000000, I.f32)
        low_result = value * low_polynomial

        half_input = I.sqrt(I.cast(0.5, I.f32) * (I.cast(1.0, I.f32) - value))
        high_y = half_input * half_input
        high_polynomial = I.cast(0.001373291015625, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.009761609511, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.011551800890, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.013964843750, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.017352764423, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.022372159090, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.030381944444, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.044642857143, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.075000000000, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(0.166666666667, I.f32)
        high_polynomial = high_polynomial * high_y + I.cast(1.000000000000, I.f32)
        high_result = I.cast(1.570796326795, I.f32) - I.cast(2.0, I.f32) * (
            half_input * high_polynomial
        )

        finite = magnitude <= I.cast(1.0, I.f32)
        selected = I.select(
            magnitude > I.cast(0.5, I.f32), high_result, low_result
        )
        valid_result = I.select(finite, selected, I.cast(I.inf, I.f32) - I.cast(I.inf, I.f32))
        result[element] = I.cast(valid_result, I.f16)


def build(context):
    kernel = context.compile("asin_elementwise", _asin_elementwise)

    def asin(input_tensor, *, out=None):
        output = torch.empty_like(input_tensor) if out is None else out
        kernel(input_tensor, output)
        return output

    return asin
