import torch
import intent
import intent.language as I


@intent.kernel
def exp_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input_tensor.shape[0]
    elements = I.domain(0, n)
    output[elements] = I.exp(input_tensor[elements])


def build(context):
    compiled_exp = context.compile("exp_float32_1d", exp_kernel)

    def exp(input_tensor, out=None):
        result = input_tensor if out is None else out
        if out is None:
            result = torch.empty_like(input_tensor)
        compiled_exp(input_tensor, result)
        return result

    return exp
