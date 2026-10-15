import torch
import intent
import intent.language as I


@intent.kernel
def _relu_sqrt(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    values = input[:]
    zero = I.cast(0.0, I.f32)
    rectified = I.select(values <= zero, zero, values)
    output[:] = I.sqrt(rectified)


def build(context):
    kernel = context.compile("relu_sqrt_pointwise", _relu_sqrt)

    def relu_sqrt(input, inplace=False, out=None):
        if inplace:
            result = input
        elif out is not None:
            result = out
        else:
            result = torch.empty_like(input)
        kernel(input, result)
        return result

    return relu_sqrt
