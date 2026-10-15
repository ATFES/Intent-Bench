import torch
import intent
import intent.language as I


@intent.kernel
def abs_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    domain = I.domain(0, input_tensor.shape[0])
    output[domain] = I.abs(input_tensor[domain])


def build(context):
    artifact = context.compile("abs_f32_1d", abs_kernel, constexprs={})

    def abs(input_tensor, out=None):
        output = out if out is not None else torch.empty_like(input_tensor)
        artifact(input_tensor, output)
        return output

    return abs
