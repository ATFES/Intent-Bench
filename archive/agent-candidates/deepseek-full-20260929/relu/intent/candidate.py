import torch

import intent
import intent.language as I


@intent.kernel
def relu_out_kernel(
    x: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("N",)],
):
    N, = x.shape
    idx = I.domain(0, N)
    y[idx] = I.maximum(x[idx], I.cast(0.0, I.f32))


@intent.kernel
def relu_inplace_kernel(
    x: I.InOut[I.f32, ("N",)],
):
    N, = x.shape
    idx = I.domain(0, N)
    x[idx] = I.maximum(x[idx], I.cast(0.0, I.f32))


def build(context):
    out_artifact = context.compile("relu_out", relu_out_kernel)
    inplace_artifact = context.compile("relu_inplace", relu_inplace_kernel)

    def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        if inplace:
            inplace_artifact(input)
            return input
        output = torch.empty_like(input)
        out_artifact(input, output)
        return output

    return relu
