import torch
import intent
import intent.language as I


@intent.kernel
def rsqrt_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    N = input.shape[0]
    elements = I.domain(0, N)
    out[elements] = I.rsqrt(input[elements])


def build(context):
    artifact = context.compile("rsqrt_kernel", rsqrt_kernel)

    def rsqrt(
        input: torch.Tensor,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return rsqrt
