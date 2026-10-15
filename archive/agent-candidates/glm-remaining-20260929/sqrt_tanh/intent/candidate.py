from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def sqrt_tanh_elementwise(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    values = input[:]
    rooted = I.sqrt(values)
    output[:] = I.tanh(rooted)


def build(context):
    elementwise = context.compile("sqrt_tanh_elementwise", sqrt_tanh_elementwise)

    def sqrt_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        elementwise(input, out)
        return out

    return sqrt_tanh
