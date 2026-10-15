import intent
import intent.language as I
import torch


@intent.kernel
def sub_gelu_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
    APPROX_TANH: I.Constexpr[bool],
):
    n = input.shape[0]
    values = I.domain(0, n)
    z = input[values] - alpha * other[values]

    half = I.cast(0.5, I.f32)
    if APPROX_TANH:
        one = I.cast(1.0, I.f32)
        cubic = I.cast(0.044715, I.f32)
        sqrt_two_over_pi = I.cast(0.7978845608028654, I.f32)
        inner = sqrt_two_over_pi * (z + cubic * z * z * z)
        result = half * z * (one + I.tanh(inner))
    else:
        sqrt_two = I.sqrt(I.cast(2.0, I.f32))
        result = half * z * I.erf(z / sqrt_two)

    out[values] = result


def build(context):
    exact = context.compile(
        "sub_gelu_exact",
        sub_gelu_kernel,
        constexprs={"APPROX_TANH": False},
    )
    tanh = context.compile(
        "sub_gelu_tanh",
        sub_gelu_kernel,
        constexprs={"APPROX_TANH": True},
    )

    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact = tanh if approximate == "tanh" else exact
        artifact(input, other, out, alpha)
        return out

    return sub_gelu
