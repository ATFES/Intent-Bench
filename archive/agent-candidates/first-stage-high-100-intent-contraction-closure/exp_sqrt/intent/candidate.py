import torch
import intent
import intent.language as I


@intent.kernel
def _exp_sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    exponentiated = I.exp(values)
    result = I.sqrt(exponentiated)
    out[elements] = result


def build(context):
    kernel = context.compile("exp_sqrt", _exp_sqrt_kernel)

    def exp_sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return exp_sqrt
