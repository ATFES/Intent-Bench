from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def sigmoid_elementwise(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    for element in I.parallel(I.domain(0, input.shape[0])):
        output[element] = I.sigmoid(input[element])


def build(context):
    sigmoid_kernel = context.compile("sigmoid_elementwise", sigmoid_elementwise)

    def sigmoid(input, *, out=None):
        result = torch.empty_like(input) if out is None else out
        sigmoid_kernel(input, result)
        return result

    return sigmoid
