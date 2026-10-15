import torch
import intent
import intent.language as I


@intent.kernel
def rsqrt_kernel(
    src: I.In[I.f32, ("N",)],
    dst: I.Out[I.f32, ("N",)],
):
    idx = I.domain(0, src.shape[0])
    dst[idx] = I.rsqrt(src[idx])


def build(context):
    artifact = context.compile("rsqrt", rsqrt_kernel)

    def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=input.dtype)
        artifact(input, out)
        return out

    return rsqrt
