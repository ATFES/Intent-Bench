import torch
import intent
import intent.language as I


@intent.kernel
def sigmoid_row_argmax(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.i64, ("M",)],
):
    rows = I.domain(0, input.shape[0])
    x = input[rows, :]
    one = I.cast(1.0, I.f32)
    sigmoid = one / (one + I.exp(-x))
    _, indices = I.arg_reduce.max(sigmoid, axis=1)
    output[rows] = I.cast(indices, I.i64)


def build(context):
    artifact = context.compile("sigmoid_row_argmax", sigmoid_row_argmax)

    def sigmoid_argmax(input, dim=None, keepdim=False):
        return artifact.run(input)

    return sigmoid_argmax
