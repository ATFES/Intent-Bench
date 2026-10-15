import torch
import intent
import intent.language as I


@intent.kernel
def _exp_sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    domain = I.domain(0, input.shape[0])
    values = input[domain]
    # Keep the two mathematical stages in source order while retaining all
    # elements as independent logical work.
    exponent = I.exp(values)
    result = I.sqrt(exponent)
    out[domain] = result


def build(context):
    kernel = context.compile("exp_sqrt_kernel", _exp_sqrt_kernel)

    def exp_sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return exp_sqrt
