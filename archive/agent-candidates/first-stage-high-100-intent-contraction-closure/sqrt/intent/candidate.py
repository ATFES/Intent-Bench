import torch
import intent
import intent.language as I


@intent.kernel
def sqrt_kernel(
    input: I.In[I.f32, ("M",)],
    out: I.Out[I.f32, ("M",)],
):
    rows = I.domain(0, input.shape[0])
    out[rows] = I.sqrt(input[rows])


def build(context):
    artifact = context.compile("sqrt_elementwise", sqrt_kernel)

    def sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return sqrt
