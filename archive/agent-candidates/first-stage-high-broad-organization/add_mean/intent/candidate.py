import torch
import intent
import intent.language as I


N = 1048576
GROUPS = 4096
CHUNK = 256


@intent.kernel
def _partial_sums(
    input: I.In[I.f32, (N,)],
    other: I.In[I.f32, (N,)],
    partials: I.Out[I.f32, (GROUPS,)],
    alpha: I.f32,
):
    source = I.domain(0, N)
    for group in I.parallel(I.domain(0, GROUPS)):
        members = source[group * CHUNK : (group + 1) * CHUNK]
        values = input[members] + other[members] * alpha
        partials[group] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


@intent.kernel
def _finish_mean(
    partials: I.In[I.f32, (GROUPS,)],
    output: I.Out[I.f32, ()],
):
    members = I.domain(0, GROUPS)
    total = I.reduce.sum(partials[members], axis=0, acc_dtype=I.f32)
    output[()] = total / I.cast(N, I.f32)


def build(context):
    partial_artifact = context.compile("add_mean_partial_sums", _partial_sums, constexprs={})
    finish_artifact = context.compile("add_mean_finish", _finish_mean, constexprs={})

    def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        if out is None:
            output = torch.empty((), device=input.device, dtype=torch.float32)
        else:
            output = out

        partials = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        partial_artifact(input, other, partials, float(alpha))
        finish_artifact(partials, output)
        return output

    return add_mean
