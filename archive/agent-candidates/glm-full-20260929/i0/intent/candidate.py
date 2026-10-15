from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _i0_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    coordinates = I.domain(0, input_tensor.shape[0])
    output[coordinates] = I.i0(input_tensor[coordinates])


def build(context):
    i0_kernel = context.compile("i0_pointwise", _i0_kernel)

    def i0(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        i0_kernel(input_tensor, out)
        return out

    return i0
