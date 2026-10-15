from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def relu_sqrt_kernel(
    source: I.In[I.f32, ("N",)],
    destination: I.Out[I.f32, ("N",)],
) -> None:
    for index in I.parallel(I.domain(0, source.shape[0])):
        destination[index] = I.sqrt(I.maximum(source[index], I.cast(0.0, I.f32)))


def build(context):
    kernel = context.compile("relu_sqrt_elementwise", relu_sqrt_kernel)

    def relu_sqrt(input, inplace: bool = False, out=None):
        if out is None:
            destination = input if inplace else allocate_output(input)
        else:
            destination = out

        kernel(input, destination)
        return destination

    return relu_sqrt


def allocate_output(tensor):
    return torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
