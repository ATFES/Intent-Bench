import torch
import intent
import intent.language as I


@intent.kernel
def _i0_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input_tensor.shape[0])
    output[elements] = I.i0(input_tensor[elements])


def build(context):
    compiled_i0 = context.compile("i0_pointwise", _i0_kernel)

    def i0(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        compiled_i0(input_tensor, out)
        return out

    return i0
