from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _abs_elementwise(
    input_tensor: I.In[I.f32, ("N",)],
    output_tensor: I.Out[I.f32, ("N",)],
) -> None:
    for element_index in I.parallel(I.domain(0, input_tensor.shape[0])):
        output_tensor[element_index] = I.abs(input_tensor[element_index])


def build(context):
    abs_kernel = context.compile("abs_elementwise", _abs_elementwise)

    def abs(input_tensor, out=None):
        output_tensor = torch.empty_like(input_tensor) if out is None else out
        abs_kernel(input_tensor, output_tensor)
        return output_tensor

    return abs
