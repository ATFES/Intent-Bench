import torch
import intent
import intent.language as I


@intent.kernel
def tanh_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output_tensor: I.Out[I.f32, ("N",)],
):
    n = input_tensor.shape[0]
    elements = I.domain(0, n)
    output_tensor[elements] = I.tanh(input_tensor[elements])


def build(context):
    artifact = context.compile("tanh_f32_1d", tanh_kernel, constexprs={})

    def tanh(input_tensor, out_tensor=None):
        if out_tensor is None:
            out_tensor = torch.empty_like(input_tensor)
        artifact(input_tensor, out_tensor)
        return out_tensor

    return tanh
