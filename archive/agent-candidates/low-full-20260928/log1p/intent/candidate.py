import torch
import intent
import intent.language as I


@intent.kernel
def log1p_kernel(
    x: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("N",)],
):
    n = x.shape[0]
    for i in I.parallel(I.domain(0, n)):
        y[i] = I.log1p(x[i])


def build(context):
    compiled = context.compile("log1p_pointwise", log1p_kernel)

    def log1p(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = out if out is not None else torch.empty_like(input)
        compiled(input, result)
        return result

    return log1p
