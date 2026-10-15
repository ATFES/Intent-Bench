from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def tanh_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    out_tensor: I.Out[I.f32, ("N",)],
) -> None:
    for element_index in I.parallel(I.domain(0, input_tensor.shape[0])):
        out_tensor[element_index] = I.tanh(input_tensor[element_index])


def build(context):
    tanh_artifact = context.compile("elementwise_tanh", tanh_kernel)

    def tanh(input_tensor, out_tensor=None):
        if out_tensor is None:
            out_tensor = torch.empty_like(input_tensor)
        tanh_artifact(input_tensor, out_tensor)
        return out_tensor

    return tanh
