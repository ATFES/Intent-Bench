import torch
import intent
import intent.language as I


@intent.kernel
def partial_max(
    input: I.In[I.f32, (1048576,)],
    values: I.Out[I.f32, (4096,)],
    indices: I.Out[I.i64, (4096,)],
):
    grouped = I.reshape(input[:], (4096, 256))
    maxima, local_indices = I.arg_reduce.max(grouped, axis=1)
    groups = I.domain(0, 4096)
    global_indices = I.indices(groups) * 256 + local_indices
    values[groups] = maxima
    indices[groups] = I.cast(global_indices, I.i64)


@intent.kernel
def finish_max(
    values: I.In[I.f32, (4096,)],
    indices: I.In[I.i64, (4096,)],
    maximum: I.Out[I.f32, ()],
    index: I.Out[I.i64, ()],
):
    value, group = I.arg_reduce.max(values[:], axis=0)
    maximum[()] = value
    index[()] = indices[group]


def build(context):
    partial = context.compile("partial_max", partial_max)
    finish = context.compile("finish_max", finish_max)

    def wrapper(input, dim, keepdim=False):
        values = torch.empty((4096,), dtype=input.dtype, device=input.device)
        indices = torch.empty((4096,), dtype=torch.int64, device=input.device)
        value = torch.empty((), dtype=input.dtype, device=input.device)
        index = torch.empty((), dtype=torch.int64, device=input.device)
        partial(input, values, indices)
        finish(values, indices, value, index)
        if keepdim:
            return value.reshape(1), index.reshape(1)
        return value, index

    return wrapper
