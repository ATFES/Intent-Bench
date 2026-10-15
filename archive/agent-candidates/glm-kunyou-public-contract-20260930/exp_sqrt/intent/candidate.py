import torch
import intent
import intent.language as I


@intent.kernel
def exp_sqrt_kernel(x: I.In[I.f32, (1048576,)], y: I.Out[I.f32, (1048576,)]):
    y[:] = I.sqrt(I.exp(x[:]))


def build(context):
    artifact = context.compile("exp_sqrt", exp_sqrt_kernel)

    def exp_sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return exp_sqrt

