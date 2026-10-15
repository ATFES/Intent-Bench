import torch
import intent
import intent.language as I


@intent.kernel
def combined_activation_kernel(
    input: I.In[I.f32, ("M", "K")],
    weight1: I.In[I.f32, ("K", "N")],
    weight2: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
):
    matmul_result = I.matmul(input, weight1, acc_dtype=I.f32)
    activated = I.tanh(I.sigmoid(matmul_result))
    result = activated * weight2 + bias
    out[...] = result


def build(context):
    kernel = context.compile(
        "combined_activation", combined_activation_kernel, constexprs={}
    )

    def combined_activation(input, weight1, weight2, bias, *, out=None):
        if out is None:
            out = torch.empty(
                (*input.shape[:-1], weight1.shape[-1]),
                device=input.device,
                dtype=input.dtype,
            )
        kernel(input, weight1, weight2, bias, out)
        return out

    return combined_activation
