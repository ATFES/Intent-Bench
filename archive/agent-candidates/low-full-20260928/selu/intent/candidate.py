import torch
import intent
import intent.language as I


@intent.kernel
def _selu_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    alpha = I.cast(1.6732632423543772, I.f32)
    scale = I.cast(1.0507009873554805, I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    for i in I.parallel(I.domain(0, input.shape[0])):
        x = input[i]
        positive = I.maximum(x, zero)
        negative = I.minimum(x, alpha * (I.exp(x) - one))
        output[i] = scale * (positive + negative)


def build(context):
    kernel = context.compile("selu_pointwise", _selu_kernel)

    def selu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        output = torch.empty_like(input)
        kernel(input, output)
        return output

    return selu
