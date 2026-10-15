import torch
import intent
import intent.language as I


@intent.kernel
def add_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    domain = I.domain(0, input.shape[0])
    for i in I.parallel(domain):
        out[i] = input[i] + alpha * other[i]


def build(context):
    artifact = context.compile("add_pointwise", add_kernel)

    def add(input, other, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, other, out, alpha)
        return out

    return add
