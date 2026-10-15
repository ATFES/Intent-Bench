import torch
import intent
import intent.language as I


_N = 1048576
_CHUNK = 1024
_GROUPS = _N // _CHUNK


@intent.kernel
def _partial_sum(
    input: I.In[I.f32, (_N,)],
    partials: I.Out[I.f32, (_GROUPS,)],
):
    groups = I.domain(0, _GROUPS)
    values = input[I.domain(0, _N)]
    chunks = I.reshape(values, (_GROUPS, _CHUNK))
    partials[groups] = I.reduce.sum(chunks, axis=1, acc_dtype=I.f32)


@intent.kernel
def _final_sum(
    partials: I.In[I.f32, (_GROUPS,)],
    output: I.Out[I.f32, ()],
):
    values = partials[I.domain(0, _GROUPS)]
    output[()] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


def build(context):
    partial_artifact = context.compile("sum_partial", _partial_sum, constexprs={})
    final_artifact = context.compile("sum_final", _final_sum, constexprs={})

    def sum(input, dim, keepdim=False, dtype=None):
        partials = torch.empty(
            (_GROUPS,), dtype=torch.float32, device=input.device
        )
        partial_artifact(input, partials)

        output = torch.empty((), dtype=torch.float32, device=input.device)
        final_artifact(partials, output)
        return output

    return sum
