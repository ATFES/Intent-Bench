import torch
import intent
import intent.language as I


@intent.kernel
def _leaky_relu_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    negative_slope: I.f32,
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        value = input[i]
        output[i] = I.select(value < 0.0, value * negative_slope, value)


def build(context):
    kernel = context.compile("leaky_relu_pointwise", _leaky_relu_kernel)

    def leaky_relu(input, negative_slope=0.01, inplace=False):
        output = input if inplace else torch.empty_like(input)
        kernel(input, output, negative_slope)
        return output

    return leaky_relu
