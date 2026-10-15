from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _mul_relu_tensor(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    elements = I.domain(0, input.shape[0])
    product = input[elements] * other[elements]
    output[elements] = I.maximum(product, I.cast(0.0, I.f32))


@intent.kernel
def _mul_relu_scalar(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    other: I.f32,
) -> None:
    elements = I.domain(0, input.shape[0])
    product = input[elements] * other
    output[elements] = I.maximum(product, I.cast(0.0, I.f32))


def build(context):
    tensor_artifact = context.compile("mul_relu_tensor", _mul_relu_tensor)
    scalar_artifact = context.compile("mul_relu_scalar", _mul_relu_scalar)

    def mul_relu(input, other, inplace=False, out=None):
        if out is None:
            output = input if inplace else torch.empty_like(input)
        else:
            output = out

        if isinstance(other, torch.Tensor):
            tensor_artifact(input, other, output)
        else:
            scalar_artifact(input, output, float(other))

        return output

    return mul_relu
