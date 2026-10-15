import torch
import intent
import intent.language as I


@intent.kernel
def abs_kernel(
    x: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("N",)],
):
    n = x.shape[0]
    idx = I.domain(0, n)
    y[idx] = I.abs(x[idx])


def build(context):
    artifact = context.compile("abs_kernel", abs_kernel)

    def abs(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        artifact(input_tensor, out)
        return out

    return abs
