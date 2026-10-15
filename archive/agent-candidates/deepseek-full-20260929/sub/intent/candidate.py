import torch

import intent
import intent.language as I


@intent.kernel
def sub_kernel(
    x: I.In[I.f32, ("N",)],
    y: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    N, = x.shape
    idx = I.domain(0, N)
    out[idx] = x[idx] - alpha * y[idx]


def build(context):
    artifact = context.compile("sub_kernel", sub_kernel)

    def sub(input, other, *, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, other, out, float(alpha))
        return out

    return sub
