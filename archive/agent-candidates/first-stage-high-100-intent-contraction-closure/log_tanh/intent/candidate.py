import torch
import intent
import intent.language as I


@intent.kernel
def _log_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    values = I.domain(0, input.shape[0])
    out[values] = I.tanh(I.log(input[values]))


def build(context):
    artifact = context.compile("log_tanh", _log_tanh_kernel, constexprs={})

    def log_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return log_tanh
