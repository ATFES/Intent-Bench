import torch
import intent
import intent.language as I


@intent.kernel
def _max_1d(
    x: I.In[I.f32, ("N",)],
    values: I.Out[I.f32, ()],
    indices: I.Out[I.i64, ()],
):
    # The arg-reduction carries the value/index pair, including first-tie
    # selection, across the complete logical reduction domain.
    value, index = I.arg_reduce.max(x[:], axis=0)
    values[()] = value
    indices[()] = I.cast(index, I.i64)


def build(context):
    kernel = context.compile("max_1d_argreduce", _max_1d)

    def max(input, dim, keepdim=False):
        values = torch.empty((), device=input.device, dtype=torch.float32)
        indices = torch.empty((), device=input.device, dtype=torch.int64)
        kernel(input, values, indices)
        return values, indices

    return max
