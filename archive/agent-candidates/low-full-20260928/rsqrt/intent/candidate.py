import torch
import intent
import intent.language as I


@intent.kernel
def rsqrt_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        output[i] = I.rsqrt(input[i])


def build(context):
    artifact = context.compile("rsqrt_pointwise", rsqrt_kernel)

    def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return rsqrt
