import torch
import intent
import intent.language as I


@intent.kernel
def _log_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    domain = I.domain(0, input.shape[0])
    for i in I.parallel(domain):
        out[i] = I.tanh(I.log(input[i]))


def build(context):
    kernel = context.compile("log_tanh_pointwise", _log_tanh_kernel)

    def log_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return log_tanh
