import torch
import intent
import intent.language as I


@intent.kernel
def _mul_relu_tensor(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    product = input[elements] * other[elements]
    output[elements] = I.maximum(product, I.cast(0.0, I.f32))


@intent.kernel
def _mul_relu_scalar(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    other: I.f32,
):
    n, = input.shape
    elements = I.domain(0, n)
    product = input[elements] * other
    output[elements] = I.maximum(product, I.cast(0.0, I.f32))


def build(context):
    tensor_kernel = context.compile("mul_relu_tensor", _mul_relu_tensor)
    scalar_kernel = context.compile("mul_relu_scalar", _mul_relu_scalar)

    def mul_relu(input, other, inplace=False, out=None):
        if inplace:
            result = input
        elif out is None:
            result = torch.empty_like(input)
        else:
            result = out

        if isinstance(other, torch.Tensor):
            tensor_kernel(input, other, result)
        else:
            scalar_kernel(input, result, other)
        return result

    return mul_relu
