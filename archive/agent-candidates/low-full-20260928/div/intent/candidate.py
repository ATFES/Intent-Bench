import torch
import intent
import intent.language as I


@intent.kernel
def div_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    rows = I.domain(0, input.shape[0])
    for i in I.parallel(rows):
        out[i] = input[i] / other[i]


def build(context):
    div_artifact = context.compile("div_elementwise", div_kernel)

    def div(input, other, rounding_mode=None, out=None):
        if out is None:
            out = torch.empty_like(input)
        div_artifact(input, other, out)
        return out

    return div
