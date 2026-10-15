import torch
import intent
import intent.language as I


GROUPS = 1024
GROUP_SIZE = 1024


@intent.kernel
def reduce_groups(
    x: I.In[I.f32, (1048576,)],
    partial: I.Out[I.f32, (GROUPS,)],
):
    groups = I.domain(0, GROUPS)
    members = I.domain(0, GROUP_SIZE)
    for group in I.parallel(groups):
        offsets = I.indices(members)
        values = x[group * GROUP_SIZE + offsets]
        partial[group] = I.reduce.sum(values, axis=0)


@intent.kernel
def reduce_partials(
    partial: I.In[I.f32, (GROUPS,)],
    result: I.Out[I.f32, ()],
):
    members = I.domain(0, GROUPS)
    values = partial[members]
    result[()] = I.reduce.sum(values, axis=0)


def build(context):
    group_artifact = context.compile("sum_reduce_groups", reduce_groups)
    final_artifact = context.compile("sum_reduce_partials", reduce_partials)

    def sum(input, dim, keepdim=False, dtype=None):
        partial = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        result = torch.empty((), device=input.device, dtype=torch.float32)
        group_artifact(input, partial)
        final_artifact(partial, result)
        return result

    return sum
