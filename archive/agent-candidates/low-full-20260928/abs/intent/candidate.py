import torch
import intent
import intent.language as I


@intent.kernel
def abs_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    domain = I.domain(0, input_tensor.shape[0])
    for index in I.parallel(domain):
        output[index] = I.abs(input_tensor[index])


def build(context):
    compiled_abs = context.compile("abs_pointwise", abs_kernel)

    def abs(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        compiled_abs(input_tensor, out)
        return out

    return abs
