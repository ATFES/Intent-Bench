import torch
import intent
import intent.language as I


@intent.kernel
def erfc_sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    erfc_out: I.Out[I.f32, ("N",)],
    sqrt_out: I.Out[I.f32, ("N",)],
):
    # Each element is independent, so both unary results share one launch.
    for i in I.parallel(I.domain(0, input.shape[0])):
        x = input[i]
        erfc_out[i] = I.erfc(x)
        sqrt_out[i] = I.sqrt(x)


def build(context):
    kernel = context.compile("erfc_sqrt_pointwise", erfc_sqrt_kernel)

    def erfc_sqrt(input: torch.Tensor):
        erfc_out = torch.empty_like(input)
        sqrt_out = torch.empty_like(input)
        kernel(input, erfc_out, sqrt_out)
        return erfc_out, sqrt_out

    return erfc_sqrt
