import torch
import intent
import intent.language as I


@intent.kernel
def _mul_relu(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        product = input[i] * other[i]
        out[i] = I.select(product > I.cast(0.0, I.f32), product, I.cast(0.0, I.f32))


def build(context):
    artifact = context.compile("mul_relu", _mul_relu)

    def mul_relu(input, other, inplace=False, out=None):
        if inplace:
            result = input
        elif out is not None:
            result = out
        else:
            result = torch.empty_like(input)
        artifact(input, other, result)
        return result

    return mul_relu
