from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _div_f32_kernel(
    numerator: I.In[I.f32, ("N",)],
    denominator: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = numerator.shape[0]
    indices = I.domain(0, n)
    output[indices] = I.fdiv(numerator[indices], denominator[indices])


def build(context):
    div_kernel = context.compile("div_f32_kernel", _div_f32_kernel)

    def div(input, other, rounding_mode=None, out=None):
        if out is None:
            out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        div_kernel(input, other, out)
        return out

    return div
