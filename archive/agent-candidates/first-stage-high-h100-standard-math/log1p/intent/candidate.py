import torch
import intent
import intent.language as I


@intent.kernel
def log1p_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    output[elements] = I.log1p(input[elements])


def build(context):
    artifact = context.compile("log1p", log1p_kernel)

    def log1p(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return log1p
