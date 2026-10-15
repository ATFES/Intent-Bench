import torch
import intent
import intent.language as I


GROUPS = 256
GROUP_SIZE = 4096
N = GROUPS * GROUP_SIZE


@intent.kernel
def partial_max(
    inp: I.In[I.f32, (N,)],
    partial: I.Out[I.f32, (GROUPS,)],
):
    values = I.reshape(inp[I.domain(0, N)], (GROUPS, GROUP_SIZE))
    maxima = I.reduce.max(values, axis=1)
    partial[I.domain(0, GROUPS)] = maxima


@intent.kernel
def finish_logsumexp(
    inp: I.In[I.f32, (N,)],
    partial: I.In[I.f32, (GROUPS,)],
    out: I.Out[I.f32, ()],
):
    values = inp[I.domain(0, N)]
    maximum = I.reduce.max(partial[I.domain(0, GROUPS)], axis=0)
    terms = I.exp(values - maximum)
    total = I.reduce.sum(terms, axis=0)
    out[()] = I.log(total) + maximum


def build(context):
    max_artifact = context.compile("logsumexp_partial_max", partial_max)
    finish_artifact = context.compile("logsumexp_finish", finish_logsumexp)

    def logsumexp(input, dim, keepdim=False, out=None):
        if out is None:
            result = torch.empty((), dtype=input.dtype, device=input.device)
        else:
            result = out
        partial = torch.empty((GROUPS,), dtype=torch.float32, device=input.device)
        max_artifact(input, partial)
        finish_artifact(input, partial, result)
        return result

    return logsumexp
