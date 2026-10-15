import torch
import intent
import intent.language as I


GROUPS = 256
CHUNK = 4096
N = GROUPS * CHUNK


@intent.kernel
def partial_stats(
    x: I.In[I.f32, ("G", "C")],
    sums: I.Out[I.f32, ("G",)],
    squares: I.Out[I.f32, ("G",)],
):
    values = x[:, :]
    sums[:] = I.reduce.sum(values, axis=1, acc_dtype=I.f32)
    squares[:] = I.reduce.sum(values * values, axis=1, acc_dtype=I.f32)


@intent.kernel
def finish_stats(
    sums: I.In[I.f32, ("G",)],
    squares: I.In[I.f32, ("G",)],
    correction: I.i64,
    out: I.Out[I.f32, ()],
):
    total = I.reduce.sum(sums, axis=0, acc_dtype=I.f32)
    total_square = I.reduce.sum(squares, axis=0, acc_dtype=I.f32)
    count = I.cast(N, I.f32)
    delta = I.cast(correction, I.f32)
    mean = total / count
    variance_numerator = total_square - total * mean
    variance = variance_numerator / (count - delta)
    out[()] = I.sqrt(variance)


def build(context):
    partial = context.compile("std_partial_stats", partial_stats)
    finish = context.compile("std_finish_stats", finish_stats)

    def std(input, dim=None, *, correction=1, keepdim=False, out=None):
        grouped = input.reshape((GROUPS, CHUNK))
        sums = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        squares = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        partial(grouped, sums, squares)
        if out is None:
            result = torch.empty((), device=input.device, dtype=input.dtype)
        else:
            result = out
        finish(sums, squares, correction, result)
        return result

    return std
