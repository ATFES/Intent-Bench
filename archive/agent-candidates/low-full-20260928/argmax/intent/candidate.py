import torch
import intent
import intent.language as I


GROUPS = 4096
GROUP_SIZE = 256
N = GROUPS * GROUP_SIZE


@intent.kernel
def reduce_groups(
    x: I.In[I.f16, (N,)],
    partial_values: I.Out[I.f16, (GROUPS,)],
    partial_indices: I.Out[I.i64, (GROUPS,)],
):
    values = I.reshape(x[I.domain(0, N)], (GROUPS, GROUP_SIZE))
    best_values, local_indices = I.arg_reduce.max(values, axis=1)
    groups = I.indices(I.domain(0, GROUPS))
    global_indices = local_indices + groups * GROUP_SIZE
    partial_values[:] = best_values
    partial_indices[:] = I.cast(global_indices, I.i64)


@intent.kernel
def reduce_partials(
    partial_values: I.In[I.f16, (GROUPS,)],
    partial_indices: I.In[I.i64, (GROUPS,)],
    result: I.Out[I.i64, ()],
):
    values = partial_values[I.domain(0, GROUPS)]
    _, winner = I.arg_reduce.max(values, axis=0)
    result[()] = partial_indices[winner]


def build(context):
    stage1 = context.compile("argmax_reduce_groups", reduce_groups)
    stage2 = context.compile("argmax_reduce_partials", reduce_partials)

    def argmax(input_tensor, dim, keepdim=False):
        partial_values = torch.empty((GROUPS,), device=input_tensor.device, dtype=input_tensor.dtype)
        partial_indices = torch.empty((GROUPS,), device=input_tensor.device, dtype=torch.int64)
        result = torch.empty((), device=input_tensor.device, dtype=torch.int64)
        stage1(input_tensor, partial_values, partial_indices)
        stage2(partial_values, partial_indices, result)
        return result

    return argmax
