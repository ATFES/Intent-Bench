import intent
import intent.language as I
import torch


@intent.kernel
def sqrt_exp_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    rows = I.domain(0, input.shape[0])
    out[rows] = I.exp(I.sqrt(input[rows]))


def build(context):
    artifact = context.compile("sqrt_exp_pointwise", sqrt_exp_kernel)

    def sqrt_exp(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return sqrt_exp
