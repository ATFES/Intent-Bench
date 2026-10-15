import intent
import intent.language as I
import torch


@intent.kernel
def sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    (N,) = input.shape
    idx = I.domain(0, N)
    out[idx] = I.sqrt(input[idx])


def build(context):
    artifact = context.compile("sqrt_kernel", sqrt_kernel)

    def sqrt(input, *, out=None):
        if out is None:
            out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        artifact(input, out)
        return out

    return sqrt
