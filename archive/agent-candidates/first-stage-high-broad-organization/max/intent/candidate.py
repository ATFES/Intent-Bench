import torch
import intent
import intent.language as I


CHUNK = 1024
GROUPS = 1024
N = CHUNK * GROUPS


@intent.kernel
def chunk_argmax(
    x: I.In[I.f32, (N,)],
    partial_values: I.Out[I.f32, (GROUPS,)],
    partial_indices: I.Out[I.i64, (GROUPS,)],
):
    groups = I.domain(0, GROUPS)
    for group in I.parallel(groups):
        begin = group * CHUNK
        end = begin + CHUNK
        chunk = x[begin:end]
        value, local_index = I.arg_reduce.max(chunk, axis=0)
        partial_values[group] = value
        partial_indices[group] = I.cast(begin + local_index, I.i64)


@intent.kernel
def final_argmax(
    partial_values: I.In[I.f32, (GROUPS,)],
    partial_indices: I.In[I.i64, (GROUPS,)],
    value_out: I.Out[I.f32, ()],
    index_out: I.Out[I.i64, ()],
):
    groups = I.domain(0, GROUPS)
    value, winning_group = I.arg_reduce.max(partial_values[groups], axis=0)
    value_out[()] = value
    index_out[()] = partial_indices[winning_group]


def build(context):
    chunk_artifact = context.compile("chunk_argmax", chunk_argmax)
    final_artifact = context.compile("final_argmax", final_argmax)

    def max_wrapper(input, dim, keepdim=False):
        partial_values = torch.empty(
            (GROUPS,), device=input.device, dtype=input.dtype
        )
        partial_indices = torch.empty(
            (GROUPS,), device=input.device, dtype=torch.int64
        )
        values = torch.empty((), device=input.device, dtype=input.dtype)
        indices = torch.empty((), device=input.device, dtype=torch.int64)

        chunk_artifact(input, partial_values, partial_indices)
        final_artifact(partial_values, partial_indices, values, indices)
        return values, indices

    return max_wrapper
