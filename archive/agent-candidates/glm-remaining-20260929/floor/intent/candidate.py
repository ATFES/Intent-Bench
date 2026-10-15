from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def floor_f32(input: I.In[I.f32, ("N",)], output: I.Out[I.f32, ("N",)]):
    positions = I.domain(0, input.shape[0])
    output[positions] = I.floor(input[positions])


def build(context):
    floor_kernel = context.compile("floor_f32", floor_f32)

    def floor(input, out=None):
        output = out
        if output is None:
            output = torch.empty_like(input)
        floor_kernel(input, output)
        return output

    return floor
