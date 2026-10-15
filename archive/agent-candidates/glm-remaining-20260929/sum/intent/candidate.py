from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def sum_partials(
    values: I.In[I.f32, ("INPUT_SIZE",)],
    partials: I.Out[I.f32, ("PARTIAL_COUNT",)],
    input_size: I.i64,
    CHUNK_SIZE: I.Constexpr[int],
) -> None:
    limit = I.cast(input_size, I.index)
    for partial_index in I.parallel(I.domain(0, partials.shape[0])):
        start = partial_index * CHUNK_SIZE
        stop = I.minimum(start + CHUNK_SIZE, limit)
        members = values[start:stop]
        partials[partial_index] = I.reduce.sum(members, axis=0, acc_dtype=I.f32)


@intent.kernel
def reduce_partials(
    partials: I.In[I.f32, ("PARTIAL_COUNT",)],
    result: I.Out[I.f32, ()],
) -> None:
    result[()] = I.reduce.sum(partials, axis=0, acc_dtype=I.f32)


def build(context):
    chunk_size = 128
    partial_sum_kernel = context.compile(
        "sum_blocked_partials_f32",
        sum_partials,
        constexprs={"CHUNK_SIZE": chunk_size},
    )
    final_sum_kernel = context.compile(
        "reduce_sum_partials_f32",
        reduce_partials,
    )

    def sum(input, dim=0, keepdim=False, dtype=None):
        partial_count = (input.numel() + chunk_size - 1) // chunk_size
        partials = torch.empty(
            (partial_count,), device=input.device, dtype=torch.float32
        )
        partial_sum_kernel(input, partials, input.numel())
        output = torch.empty(
            (), device=input.device, dtype=torch.float32
        )
        final_sum_kernel(partials, output)
        return output

    return sum
