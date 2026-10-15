import torch
import intent
import intent.language as I


@intent.kernel
def _ifftshift_1d(
    x: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("N",)],
):
    n = x.shape[0]
    shift = (n + 1) // 2
    for i in I.parallel(I.domain(0, n)):
        y[i] = x[(i + shift) % n]


def build(context):
    kernel = context.compile("ifftshift_1d", _ifftshift_1d)

    def ifftshift(input, dim=None):
        output = torch.empty_like(input)
        kernel(input, output)
        return output

    return ifftshift
