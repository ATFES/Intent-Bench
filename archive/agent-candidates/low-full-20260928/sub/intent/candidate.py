import torch
import intent
import intent.language as I


@intent.kernel
def _sub_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        out[i] = input[i] - alpha * other[i]


def build(context):
    kernel = context.compile("sub_pointwise", _sub_kernel)

    def sub(input, other, *, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, other, out, alpha)
        return out

    return sub
