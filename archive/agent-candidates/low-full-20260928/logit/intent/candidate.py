import torch
import intent
import intent.language as I


@intent.kernel
def _logit_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    eps_value: I.f32,
    CLAMP: I.Constexpr[bool],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        x = input[i]
        if CLAMP:
            x = I.select(x < eps_value, eps_value, x)
            upper = I.cast(1.0, I.f32) - eps_value
            x = I.select(x > upper, upper, x)
        out[i] = I.log(x / (I.cast(1.0, I.f32) - x))


def build(context):
    unclamped = context.compile("logit_unclamped", _logit_kernel, constexprs={"CLAMP": False})
    clamped = context.compile("logit_clamped", _logit_kernel, constexprs={"CLAMP": True})

    def logit(input: torch.Tensor, eps: float = None, out: torch.Tensor = None):
        result = out if out is not None else torch.empty_like(input)
        eps_value = 0.0 if eps is None else float(eps)
        artifact = unclamped if eps is None else clamped
        artifact(input, result, eps_value)
        return result

    return logit
