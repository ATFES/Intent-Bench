from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def _tanh_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    out_tensor: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, input_tensor.shape[0])):
        out_tensor[i] = I.tanh(input_tensor[i])


def build(context):
    tanh_artifact = context.compile("tanh_kernel", _tanh_kernel, constexprs={})

    def tanh(input_tensor, out_tensor=None):
        if out_tensor is None:
            out_tensor = torch.empty(
                input_tensor.shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

        tanh_artifact(input_tensor, out_tensor)
        return out_tensor

    return tanh
