import math

import intent
import intent.language as I


INV_SQRT_2 = 1.0 / math.sqrt(2.0)
TANH_SCALE = math.sqrt(2.0 / math.pi)
TANH_COEFF = 0.044715


@intent.kernel
def sub_gelu_erf(
    output: I.Out[I.f32, ("N",)],
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    alpha: I.f32,
):
    z = input - alpha * other
    erf_value = I.erf(I.fdiv(z, I.cast(INV_SQRT_2, I.f32)))
    output[:] = 0.5 * z * erf_value


@intent.kernel
def sub_gelu_tanh(
    output: I.Out[I.f32, ("N",)],
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    alpha: I.f32,
):
    z = input - alpha * other
    cubic = z * z * z
    tanh_value = I.tanh(I.cast(TANH_SCALE, I.f32) * (z + TANH_COEFF * cubic))
    output[:] = 0.5 * z * (1.0 + tanh_value)


def build(context):
    erf_artifact = context.compile("sub_gelu_erf", sub_gelu_erf)
    tanh_artifact = context.compile("sub_gelu_tanh", sub_gelu_tanh)

    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        if approximate == "tanh":
            artifact = tanh_artifact
        else:
            artifact = erf_artifact

        if out is None:
            return artifact.run(input, other, alpha)
        artifact(out, input, other, alpha)
        return out

    return sub_gelu
