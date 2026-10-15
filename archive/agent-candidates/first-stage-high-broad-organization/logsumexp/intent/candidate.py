import torch
import intent
import intent.language as I


CHUNKS = 4096
CHUNK_SIZE = 256
INPUT_SIZE = CHUNKS * CHUNK_SIZE


@intent.kernel
def _partial_logsumexp(
    input: I.In[I.f32, (INPUT_SIZE,)],
    partial_max: I.Out[I.f32, (CHUNKS,)],
    partial_sum: I.Out[I.f32, (CHUNKS,)],
):
    source = I.domain(0, INPUT_SIZE)
    for chunk in I.parallel(I.domain(0, CHUNKS)):
        start = chunk * CHUNK_SIZE
        members = source[start : start + CHUNK_SIZE]
        values = input[members]
        maximum = I.reduce.max(values, axis=0, acc_dtype=I.f32)

        pos_inf = I.cast(I.inf, I.f32)
        neg_inf = -pos_inf
        zero = I.cast(0.0, I.f32)
        delta = I.select(maximum == pos_inf, zero, values - maximum)
        delta = I.select(maximum == neg_inf, zero, delta)
        total = I.reduce.sum(I.exp(delta), axis=0, acc_dtype=I.f32)

        partial_max[chunk] = maximum
        partial_sum[chunk] = total


@intent.kernel
def _finalize_logsumexp(
    partial_max: I.In[I.f32, (CHUNKS,)],
    partial_sum: I.In[I.f32, (CHUNKS,)],
    output: I.Out[I.f32, ()],
):
    chunks = I.domain(0, CHUNKS)
    maxima = partial_max[chunks]
    sums = partial_sum[chunks]
    maximum = I.reduce.max(maxima, axis=0, acc_dtype=I.f32)

    pos_inf = I.cast(I.inf, I.f32)
    neg_inf = -pos_inf
    zero = I.cast(0.0, I.f32)
    delta = I.select(maximum == pos_inf, zero, maxima - maximum)
    delta = I.select(maximum == neg_inf, zero, delta)
    total = I.reduce.sum(sums * I.exp(delta), axis=0, acc_dtype=I.f32)
    output[()] = I.log(total) + maximum


def build(context):
    partial = context.compile("logsumexp_partial", _partial_logsumexp, constexprs={})
    finalize = context.compile("logsumexp_finalize", _finalize_logsumexp, constexprs={})

    def logsumexp(input, dim, keepdim=False, out=None):
        partial_max = torch.empty((CHUNKS,), device=input.device, dtype=torch.float32)
        partial_sum = torch.empty((CHUNKS,), device=input.device, dtype=torch.float32)
        output = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)
        partial(input, partial_max, partial_sum)
        finalize(partial_max, partial_sum, output)
        return output

    return logsumexp
