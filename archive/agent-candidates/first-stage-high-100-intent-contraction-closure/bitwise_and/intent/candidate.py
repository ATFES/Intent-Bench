import torch
import intent
import intent.language as I


@intent.kernel
def bitwise_and_kernel(
    input: I.In[I.i32, ("N",)],
    other: I.In[I.i32, ("N",)],
    out: I.Out[I.i32, ("N",)],
):
    n = input.shape[0]
    indices = I.domain(0, n)
    out[indices] = input[indices] & other[indices]


def build(context):
    artifact = context.compile("bitwise_and_int32", bitwise_and_kernel)

    def bitwise_and(input, other, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, other, out)
        return out

    return bitwise_and
