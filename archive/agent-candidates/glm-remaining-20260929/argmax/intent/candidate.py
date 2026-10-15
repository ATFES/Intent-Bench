import torch

import intent
import intent.language as I


GROUP_SIZE = 1024


@intent.kernel
def partial_argmax(
    input_tensor: I.In[I.f16, ("N",)],
    partial_values: I.Out[I.f16, ("GROUPS",)],
    partial_indices: I.Out[I.i64, ("GROUPS",)],
    GROUP_SIZE: I.Constexpr[int],
) -> None:
    source_axis = I.domain(0, input_tensor.shape[0])
    for group in I.parallel(I.domain(0, partial_values.shape[0])):
        begin = group * GROUP_SIZE
        end = begin + GROUP_SIZE
        members = source_axis[begin:end]
        group_values = input_tensor[members]
        best_value, best_local_index = I.arg_reduce.max(group_values, axis=0)
        partial_values[group] = best_value
        partial_indices[group] = I.cast(begin + best_local_index, I.i64)


@intent.kernel
def merge_argmax(
    partial_values: I.In[I.f16, ("GROUPS",)],
    partial_indices: I.In[I.i64, ("GROUPS",)],
    result: I.Out[I.i64, ()],
) -> None:
    best_value, best_group = I.arg_reduce.max(partial_values, axis=0)
    result[()] = partial_indices[best_group]


def build(context):
    partial_artifact = context.compile(
        "argmax_partial_groups",
        partial_argmax,
        constexprs={"GROUP_SIZE": GROUP_SIZE},
    )
    merge_artifact = context.compile("argmax_merge_groups", merge_argmax)

    def argmax(input_tensor, dim, keepdim=False):
        if input_tensor.ndim != 1 or dim not in (None, 0):
            raise ValueError("candidate is specialized to dim=0 on a 1-D tensor")
        if input_tensor.shape[0] % GROUP_SIZE != 0:
            raise ValueError("input length must be divisible by the partial group size")

        partial_values = torch.empty(
            (input_tensor.shape[0] // GROUP_SIZE,),
            dtype=torch.float16,
            device=input_tensor.device,
        )
        partial_indices = torch.empty(
            (input_tensor.shape[0] // GROUP_SIZE,),
            dtype=torch.int64,
            device=input_tensor.device,
        )
        partial_artifact(input_tensor, partial_values, partial_indices)
        return merge_artifact.run(partial_values, partial_indices)

    return argmax
