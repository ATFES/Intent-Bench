import torch
import intent
import intent.language as I


@intent.kernel
def _relu_out_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    zero = I.cast(0.0, I.f32)
    output[elements] = I.maximum(input[elements], zero)


@intent.kernel
def _relu_inplace_kernel(
    input: I.InOut[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    zero = I.cast(0.0, I.f32)
    input[elements] = I.maximum(input[elements], zero)


def build(context):
    out_kernel = context.compile("relu_out_of_place", _relu_out_kernel, constexprs={})
    inplace_kernel = context.compile("relu_in_place", _relu_inplace_kernel, constexprs={})

    def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        if inplace:
            inplace_kernel(input)
            return input

        output = torch.empty_like(input)
        out_kernel(input, output)
        return output

    return relu
