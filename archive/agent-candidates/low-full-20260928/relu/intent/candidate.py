import torch
import intent
import intent.language as I


@intent.kernel
def relu_out(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    zero = I.cast(0.0, I.f32)
    for i in I.parallel(I.domain(0, n)):
        value = input[i]
        output[i] = I.select(value > zero, value, zero)


@intent.kernel
def relu_inplace(
    input: I.InOut[I.f32, ("N",)],
):
    n = input.shape[0]
    zero = I.cast(0.0, I.f32)
    for i in I.parallel(I.domain(0, n)):
        value = input[i]
        input[i] = I.select(value > zero, value, zero)


def build(context):
    out_kernel = context.compile("relu_out", relu_out)
    inplace_kernel = context.compile("relu_inplace", relu_inplace)

    def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        if inplace:
            inplace_kernel(input)
            return input
        output = torch.empty_like(input)
        out_kernel(input, output)
        return output

    return relu
