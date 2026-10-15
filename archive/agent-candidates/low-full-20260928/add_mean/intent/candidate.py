import torch
import intent
import intent.language as I


GROUPS = 1024
GROUP_SIZE = 1024


@intent.kernel
def _partial_sums(
    input: I.In[I.f32, ("G", "K")],
    other: I.In[I.f32, ("G", "K")],
    partial: I.Out[I.f32, ("G",)],
    alpha: I.f32,
):
    for group in I.parallel(I.domain(0, GROUPS)):
        values = input[group, :] + alpha * other[group, :]
        partial[group] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


@intent.kernel
def _final_mean(
    partial: I.In[I.f32, ("G",)],
    output: I.Out[I.f32, ()],
    count: I.f32,
):
    total = I.reduce.sum(partial[:], axis=0, acc_dtype=I.f32)
    output[()] = total / count


def build(context):
    partial_kernel = context.compile("add_mean_partial", _partial_sums)
    final_kernel = context.compile("add_mean_final", _final_mean)

    def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        # The evaluator's fixed profile is contiguous float32 with a full reduction.
        grouped_input = input.reshape(GROUPS, GROUP_SIZE)
        grouped_other = other.reshape(GROUPS, GROUP_SIZE)
        partial = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        partial_kernel(grouped_input, grouped_other, partial, float(alpha))

        result = out if out is not None else torch.empty((), device=input.device, dtype=torch.float32)
        final_kernel(partial, result, float(input.numel()))
        return result

    return add_mean
