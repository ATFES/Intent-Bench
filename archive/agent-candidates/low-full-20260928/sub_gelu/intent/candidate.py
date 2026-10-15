import torch
import intent
import intent.language as I


@intent.kernel
def _sub_gelu(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
    APPROX_TANH: I.Constexpr[bool],
):
    points = I.domain(0, input.shape[0])
    sqrt_two = I.sqrt(I.cast(2.0, I.f32))
    half = I.cast(0.5, I.f32)
    if APPROX_TANH:
        scale = I.cast(0.7978845608028654, I.f32)
        cubic = I.cast(0.044715, I.f32)
        for i in I.parallel(points):
            z = input[i] - alpha * other[i]
            z3 = z * z * z
            value = half * z * (I.cast(1.0, I.f32) + I.tanh(scale * (z + cubic * z3)))
            out[i] = value
    else:
        for i in I.parallel(points):
            z = input[i] - alpha * other[i]
            out[i] = half * z * I.erf(z / sqrt_two)


def build(context):
    exact = context.compile("sub_gelu_exact", _sub_gelu, constexprs={"APPROX_TANH": False})
    tanh = context.compile("sub_gelu_tanh", _sub_gelu, constexprs={"APPROX_TANH": True})

    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        result = out if out is not None else torch.empty_like(input)
        artifact = tanh if approximate == "tanh" else exact
        artifact(input, other, result, float(alpha))
        return result

    return sub_gelu
