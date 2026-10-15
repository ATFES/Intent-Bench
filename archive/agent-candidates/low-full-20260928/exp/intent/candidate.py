import torch
import intent
import intent.language as I


@intent.kernel
def exp_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    domain = I.domain(0, input_tensor.shape[0])
    for i in I.parallel(domain):
        output[i] = I.exp(input_tensor[i])


def build(context):
    compiled_exp = context.compile("exp_pointwise", exp_kernel)

    def exp(input_tensor, out=None):
        output = out if out is not None else torch.empty_like(input_tensor)
        compiled_exp(input_tensor, output)
        return output

    return exp
