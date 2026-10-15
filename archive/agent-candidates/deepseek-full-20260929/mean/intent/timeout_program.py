import torch

import intent
import intent.language as I


GROUP_SIZE = 1024


@intent.kernel
def partial_sum(
    x: I.In[I.f32, ("R", "C")],
    partial: I.Out[I.f32, ("R",)],
):
    R, C = x.shape
    rows = I.domain(0, R)
    cols = I.domain(0, C)
    partial[rows] = I.reduce.sum(x[rows, cols], axis=1, acc_dtype=I.f32)


@intent.kernel
def final_reduce(
    partial: I.In[I.f32, ("R",)],
    count: I.f32,
    out: I.Out[I.f32, ()],
):
    R, = partial.shape
    members = I.domain(0, R)
    total = I.reduce.sum(partial[members], axis=0, acc_dtype=I.f32)
    out[()] = total / count


def build(context):
    partial_artifact = context.compile("mean_partial_sum", partial_sum)
    final_artifact = context.compile("mean_final_reduce", final_reduce)

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        n = input_tensor.shape[0]
        c = GROUP_SIZE
        r = n // c
        x2d = input_tensor.view(r, c)

        partial = torch.empty((r,), dtype=torch.float32, device=input_tensor.device)
        partial_artifact(x2d, partial)

        if out is None:
            out = torch.empty((), dtype=torch.float32, device=input_tensor.device)

        final_artifact(partial, float(n), out)
        return out

    return mean
