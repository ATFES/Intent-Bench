import torch
import intent
import intent.language as I


N = 1048576
GROUPS = 4096
CHUNK = 256


@intent.kernel
def exp_mean_stage1(
    input: I.In[I.f32, (N,)],
    partials: I.Out[I.f32, (GROUPS,)],
):
    values = input[I.domain(0, N)]
    exponentials = I.exp(values)
    grouped = I.reshape(exponentials, (GROUPS, CHUNK))
    sums = I.reduce.sum(grouped, axis=1, acc_dtype=I.f32)
    partials[I.domain(0, GROUPS)] = sums


@intent.kernel
def exp_mean_stage2(
    partials: I.In[I.f32, (GROUPS,)],
    output: I.Out[I.f32, ()],
):
    values = partials[I.domain(0, GROUPS)]
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    output[()] = total / I.cast(N, I.f32)


def build(context):
    stage1 = context.compile("exp_mean_stage1", exp_mean_stage1)
    stage2 = context.compile("exp_mean_stage2", exp_mean_stage2)

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        partials = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        result = out if out is not None else torch.empty((), device=input.device, dtype=torch.float32)
        stage1(input, partials)
        stage2(partials, result)
        return result

    return exp_mean
