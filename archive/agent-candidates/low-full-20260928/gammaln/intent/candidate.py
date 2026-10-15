import torch
import intent
import intent.language as I


@intent.kernel
def _gammaln_kernel(
    x: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    N = x.shape[0]
    for i in I.parallel(I.domain(0, N)):
        out[i] = I.lgamma(x[i])


def build(context):
    artifact = context.compile("gammaln_pointwise", _gammaln_kernel)

    def gammaln(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = out if out is not None else torch.empty_like(input)
        artifact(input, result)
        return result

    return gammaln
