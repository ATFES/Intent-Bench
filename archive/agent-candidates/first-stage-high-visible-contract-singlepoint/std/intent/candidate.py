import torch
import intent
import intent.language as I


GROUPS = 256
CHUNK = 4096
N = GROUPS * CHUNK


@intent.kernel
def _partial_std(
    x: I.In[I.f32, (N,)],
    partials: I.Out[I.f32, (GROUPS, 2)],
):
    source = I.domain(0, N)
    for group in I.parallel(I.domain(0, GROUPS)):
        begin = group * CHUNK
        end = begin + CHUNK
        members = source[begin:end]
        values = x[members]
        total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
        squares = values * values
        total_sq = I.reduce.sum(squares, axis=0, acc_dtype=I.f32)
        partials[group, 0] = total
        partials[group, 1] = total_sq


@intent.kernel
def _finish_std(
    partials: I.In[I.f32, (GROUPS, 2)],
    result: I.Out[I.f32, ()],
    correction: I.i64,
):
    totals = I.reduce.sum(partials, axis=0, acc_dtype=I.f32)
    total = totals[0]
    total_sq = totals[1]

    count = I.cast(N, I.f32)
    mean = I.fdiv(total, count)
    centered_sum = total_sq - total * mean

    correction_f = I.cast(correction, I.f32)
    denominator = I.maximum(count - correction_f, I.cast(0.0, I.f32))
    variance = I.fdiv(centered_sum, denominator)
    result[()] = I.sqrt(I.maximum(variance, I.cast(0.0, I.f32)))


def build(context):
    partial_artifact = context.compile("std_partial", _partial_std, constexprs={})
    finish_artifact = context.compile("std_finish", _finish_std, constexprs={})

    def std(input, dim=None, *, correction=1, keepdim=False, out=None):
        partials = torch.empty(
            (GROUPS, 2), dtype=torch.float32, device=input.device
        )
        partial_artifact(input, partials)

        result = out
        if result is None:
            result = torch.empty((), dtype=torch.float32, device=input.device)
        finish_artifact(partials, result, correction)
        return result

    return std
