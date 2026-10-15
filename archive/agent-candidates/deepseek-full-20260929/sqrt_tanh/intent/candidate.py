import torch

import intent
import intent.language as I


@intent.kernel
def sqrt_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
) -> None:
    N, = input.shape
    rows = I.domain(0, N)
    out[rows] = I.tanh(I.sqrt(input[rows]))


def build(context):
    sqrt_tanh_artifact = context.compile("sqrt_tanh", sqrt_tanh_kernel)

    def sqrt_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        sqrt_tanh_artifact(input, out)
        return out

    return sqrt_tanh
