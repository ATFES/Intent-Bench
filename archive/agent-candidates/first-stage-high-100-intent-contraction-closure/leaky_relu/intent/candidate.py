import torch
import intent
import intent.language as I


@intent.kernel
def leaky_relu_out(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    negative_slope: I.f32,
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    output[elements] = I.maximum(values, 0.0) + negative_slope * I.minimum(values, 0.0)


@intent.kernel
def leaky_relu_inplace(
    input: I.InOut[I.f32, ("N",)],
    negative_slope: I.f32,
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    input[elements] = I.maximum(values, 0.0) + negative_slope * I.minimum(values, 0.0)


def build(context):
    out_artifact = context.compile("leaky_relu_out", leaky_relu_out)
    inplace_artifact = context.compile("leaky_relu_inplace", leaky_relu_inplace)

    def leaky_relu(input, negative_slope=0.01, inplace=False):
        if inplace:
            inplace_artifact(input, negative_slope)
            return input
        output = torch.empty_like(input)
        out_artifact(input, output, negative_slope)
        return output

    return leaky_relu
