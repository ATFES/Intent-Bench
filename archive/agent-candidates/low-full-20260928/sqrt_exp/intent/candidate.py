import torch
import intent
import intent.language as I


@intent.kernel
def _sqrt_exp_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    domain = I.domain(0, input.shape[0])
    for i in I.parallel(domain):
        value = input[i]
        rooted = I.sqrt(value)
        out[i] = I.exp(rooted)


def build(context):
    kernel = context.compile("sqrt_exp_pointwise", _sqrt_exp_kernel)

    def sqrt_exp(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return sqrt_exp
