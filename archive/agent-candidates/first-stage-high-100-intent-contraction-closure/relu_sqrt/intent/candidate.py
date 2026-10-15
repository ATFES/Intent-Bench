import torch
import intent
import intent.language as I


@intent.kernel
def relu_sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    nonnegative = I.maximum(values, I.cast(0.0, I.f32))
    output[elements] = I.sqrt(nonnegative)


def build(context):
    artifact = context.compile("relu_sqrt_kernel", relu_sqrt_kernel, constexprs={})

    def relu_sqrt(input, inplace=False, out=None):
        if inplace:
            result = input
        elif out is not None:
            result = out
        else:
            result = torch.empty_like(input)
        artifact(input, result)
        return result

    return relu_sqrt
