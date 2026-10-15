import torch
import intent
import intent.language as I


@intent.kernel
def _max_1d(
    input: I.In[I.f32, ("N",)],
    values: I.Out[I.f32, ()],
    indices: I.Out[I.i64, ()],
):
    value, index = I.arg_reduce.max(input[:], axis=0)
    values[()] = value
    indices[()] = I.cast(index, I.i64)


def build(context):
    artifact = context.compile("max_1d", _max_1d)

    def max(input, dim, keepdim=False):
        values = torch.empty((), dtype=input.dtype, device=input.device)
        indices = torch.empty((), dtype=torch.int64, device=input.device)
        artifact(input, values, indices)
        return values, indices

    return max
