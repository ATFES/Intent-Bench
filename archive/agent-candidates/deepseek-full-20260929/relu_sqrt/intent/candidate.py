from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def _relu_sqrt_kernel(
    src: I.In[I.f32, ("N",)],
    dst: I.Out[I.f32, ("N",)],
):
    n = src.shape[0]
    zero = I.cast(0.0, I.f32)
    for i in I.parallel(I.domain(0, n)):
        x = src[i]
        dst[i] = I.sqrt(I.maximum(x, zero))


def build(context):
    relu_sqrt_kernel = context.compile("relu_sqrt", _relu_sqrt_kernel)

    def relu_sqrt(input, inplace=False, out=None):
        if inplace:
            relu_sqrt_kernel(input, input)
            return input

        if out is None:
            out = torch.empty_like(input)

        relu_sqrt_kernel(input, out)
        return out

    return relu_sqrt
