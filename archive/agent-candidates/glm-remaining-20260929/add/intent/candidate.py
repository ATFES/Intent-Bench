import intent
import intent.language as I
import torch


@intent.kernel
def _add_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    alpha: I.f32,
) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        output[index] = input[index] + alpha * other[index]


def build(context):
    add_kernel = context.compile("add_f32_contiguous", _add_kernel)

    def add(input, other, alpha=1, out=None):
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out
        add_kernel(input, other, output, alpha)
        return output

    return add
