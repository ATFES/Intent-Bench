import torch
import intent
import intent.language as I


@intent.kernel
def _log_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        out[i] = I.log(input[i])


def build(context):
    log_kernel = context.compile("log_pointwise", _log_kernel)

    def log(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        log_kernel(input, out)
        return out

    return log
