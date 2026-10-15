from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def combined_activation(
    input: I.In[I.f32, ("M", "K")],
    weight1: I.In[I.f32, ("K", "N")],
    weight2: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
) -> None:
    preactivation = I.matmul(input[:, :], weight1[:, :], acc_dtype=I.f32)
    activation = I.tanh(I.sigmoid(preactivation), approximate=False)
    out[:, :] = activation * weight2[:] + bias[:]


def build(context):
    kernel = context.compile("combined_activation_f32", combined_activation)

    def combined_activation_wrapper(input, weight1, weight2, bias, *, out=None):
        result = out if out is not None else torch.empty_like(input)
        kernel(input, weight1, weight2, bias, result)
        return result

    return combined_activation_wrapper
