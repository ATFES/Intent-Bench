import torch
import intent
import intent.language as I


@intent.kernel
def _bitwise_and(
    input: I.In[I.i32, ("N",)],
    other: I.In[I.i32, ("N",)],
    out: I.Out[I.i32, ("N",)],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        out[i] = input[i] & other[i]


def build(context):
    bitwise_and_kernel = context.compile("bitwise_and", _bitwise_and)

    def bitwise_and(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)
        bitwise_and_kernel(input, other, out)
        return out

    return bitwise_and
