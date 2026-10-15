import torch
import intent
import intent.language as I


@intent.kernel
def _add_gelu(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
    alpha: I.f32,
    APPROX: I.Constexpr[bool],
):
    n = input.shape[0]
    sqrt_two = I.cast(1.4142135623730951, I.f32)
    inv_sqrt_two = I.cast(0.7071067811865475, I.f32)
    sqrt_2_over_pi = I.cast(0.7978845608028654, I.f32)
    half = I.cast(0.5, I.f32)
    one = I.cast(1.0, I.f32)
    cubic = I.cast(0.044715, I.f32)

    for i in I.parallel(I.domain(0, n)):
        x = I.cast(input[i], I.f32) + alpha * I.cast(other[i], I.f32)
        if APPROX:
            x3 = x * x * x
            y = half * x * (one + I.tanh(sqrt_2_over_pi * (x + cubic * x3)))
        else:
            y = half * x * (one + I.erf(x * inv_sqrt_two))
        output[i] = I.cast(y, I.f16)


def build(context):
    exact = context.compile("add_gelu_exact", _add_gelu, constexprs={"APPROX": False})
    tanh = context.compile("add_gelu_tanh", _add_gelu, constexprs={"APPROX": True})

    def add_gelu(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact = tanh if approximate == "tanh" else exact
        artifact(input, other, out, float(alpha))
        return out

    return add_gelu
