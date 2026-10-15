from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def fused_tile_exp_1d(
    source: I.In[I.f32, ("SOURCE_SIZE",)],
    destination: I.Out[I.f32, ("REPEATS", "SOURCE_SIZE")],
) -> None:
    for repeat in I.parallel(I.domain(0, destination.shape[0])):
        for source_index in I.parallel(I.domain(0, destination.shape[1])):
            destination[repeat, source_index] = I.exp(source[source_index])


def build(context):
    kernel = context.compile("fused_tile_exp_1d", fused_tile_exp_1d)

    def fused_tile_exp(input, dims, *, out=None):
        repeats = tuple(dims)
        if len(input.shape) != 1:
            raise ValueError("fused_tile_exp_1d requires a one-dimensional input")
        if len(repeats) == 0:
            repeats = (1,)
        elif len(repeats) > 1:
            raise ValueError("dims has more dimensions than input")

        if out is None:
            out = torch.empty(repeats + input.shape, dtype=input.dtype, device=input.device)

        kernel(input, out)
        return out

    return fused_tile_exp
