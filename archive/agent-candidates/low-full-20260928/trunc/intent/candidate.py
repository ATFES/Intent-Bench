import torch
import intent
import intent.language as I


@intent.kernel
def trunc_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    rows = I.domain(0, input.shape[0])
    values = input[rows]
    truncated = I.select(values >= 0.0, I.floor(values), -I.floor(-values))
    output[rows] = truncated


def build(context):
    compiled = context.compile("trunc_pointwise", trunc_kernel)

    def trunc(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = out if out is not None else torch.empty_like(input)
        compiled(input, result)
        return result

    return trunc
