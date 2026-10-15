import torch
import intent
import intent.language as I


@intent.kernel
def mul_kernel(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    out: I.Out[I.f16, ("N",)],
):
    # Each logical coordinate is independent, matching the elementwise stage.
    for i in I.parallel(I.domain(0, input.shape[0])):
        out[i] = input[i] * other[i]


def build(context):
    artifact = context.compile("mul_elementwise", mul_kernel)

    def mul(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, other, out)
        return out

    return mul
