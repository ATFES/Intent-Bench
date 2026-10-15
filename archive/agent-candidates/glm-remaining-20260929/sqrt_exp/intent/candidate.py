from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def sqrt_exp_kernel(
    source: I.In[I.f32, ("N",)],
    destination: I.Out[I.f32, ("N",)],
) -> None:
    for element_id in I.parallel(I.domain(0, source.shape[0])):
        square_root = I.sqrt(source[element_id])
        destination[element_id] = I.exp(square_root)


def build(context):
    kernel = context.compile("sqrt_exp_elementwise", sqrt_exp_kernel)

    def sqrt_exp(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return sqrt_exp
