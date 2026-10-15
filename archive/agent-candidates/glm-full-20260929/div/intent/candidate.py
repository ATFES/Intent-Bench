from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def true_divide_f32(
    dividend: I.In[I.f32, ("N",)],
    divisor: I.In[I.f32, ("N",)],
    quotient: I.Out[I.f32, ("N",)],
) -> None:
    for index in I.parallel(I.domain(0, dividend.shape[0])):
        quotient[index] = dividend[index] / divisor[index]


def build(context):
    divide = context.compile("div_f32_elementwise", true_divide_f32)

    def div(input, other, rounding_mode=None, out=None):
        if rounding_mode is not None:
            raise ValueError("rounding_mode is not supported by this profile")
        if out is None:
            out = torch.empty_like(input)
        divide(input, other, out)
        return out

    return div
