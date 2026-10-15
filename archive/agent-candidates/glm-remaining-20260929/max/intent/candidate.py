from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _chunk_argmax(
    input: I.In[I.f32, ("N",)],
    chunk_values: I.Out[I.f32, (4096,)],
    CHUNK: I.Constexpr[int],
) -> None:
    columns = I.domain(0, input.shape[0])
    for group_id in I.parallel(I.domain(0, chunk_values.shape[0])):
        start = group_id * CHUNK
        stop = start + CHUNK
        chunk = columns[start:stop]
        chunk_value, _ = I.arg_reduce.max(input[chunk], axis=0)
        chunk_values[group_id] = chunk_value


@intent.kernel
def _merge_argmax(
    max_value: I.Out[I.f32, ()],
    max_index: I.Out[I.i64, ()],
    chunk_values: I.In[I.f32, (4096,)],
    CHUNK: I.Constexpr[int],
) -> None:
    winning_value, winning_group = I.arg_reduce.max(chunk_values, axis=0)
    max_value[()] = winning_value
    max_index[()] = I.cast(winning_group * CHUNK, I.i64)


def build(context):
    input_size = 1048576
    group_count = 4096
    chunk_size = input_size // group_count

    chunk_argmax = context.compile(
        "max_chunk_argmax",
        _chunk_argmax,
        constexprs={"CHUNK": chunk_size},
    )
    merge_argmax = context.compile(
        "max_merge_argmax",
        _merge_argmax,
        constexprs={"CHUNK": chunk_size},
    )

    def max(input, dim, keepdim=False):
        if input.shape != (input_size,) or dim != 0:
            raise ValueError("compiled profile requires a 1-D input reduced along dim 0")
        if input.dtype != torch.float32 or not input.is_cuda:
            raise ValueError("compiled profile requires a CUDA float32 input")

        chunk_values = chunk_argmax.run(input)
        value, index = merge_argmax.run(chunk_values)
        if keepdim:
            return value.reshape(1), index.reshape(1)
        return value, index

    return max
