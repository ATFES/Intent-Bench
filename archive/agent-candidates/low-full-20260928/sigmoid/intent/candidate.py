import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    one = I.cast(1.0, I.f32)
    for i in I.parallel(I.domain(0, n)):
        x = input[i]
        output[i] = I.fdiv(one, one + I.exp(-x))


def build(context):
    kernel = context.compile("sigmoid_elementwise", _sigmoid_kernel)

    def sigmoid(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return sigmoid
