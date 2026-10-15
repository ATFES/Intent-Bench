import torch
import intent
import intent.language as I


@intent.kernel
def sigmoid_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    out[elements] = I.sigmoid(input[elements])


def build(context):
    compiled = context.compile("sigmoid_elementwise", sigmoid_kernel, constexprs={})

    def sigmoid(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        compiled(input, out)
        return out

    return sigmoid
