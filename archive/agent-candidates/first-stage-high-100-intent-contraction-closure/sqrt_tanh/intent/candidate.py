import torch
import intent
import intent.language as I


@intent.kernel
def sqrt_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]
    out[elements] = I.tanh(I.sqrt(values))


def build(context):
    artifact = context.compile("sqrt_tanh_elementwise", sqrt_tanh_kernel, constexprs={})

    def sqrt_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return sqrt_tanh
