import torch
import intent
import intent.language as I


@intent.kernel
def sub_gelu_erf_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    idx = I.domain(0, n)
    half = I.cast(0.5, I.f32)
    rsqrt2 = I.cast(0.7071067811865476, I.f32)
    for i in I.parallel(idx):
        z = input[i] - alpha * other[i]
        out[i] = half * z * I.erf(z * rsqrt2)


@intent.kernel
def sub_gelu_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    idx = I.domain(0, n)
    half = I.cast(0.5, I.f32)
    beta = I.cast(0.7978845608028654, I.f32)
    kappa = I.cast(0.044715, I.f32)
    one = I.cast(1.0, I.f32)
    for i in I.parallel(idx):
        z = input[i] - alpha * other[i]
        inner = beta * (z + kappa * (z * z * z))
        out[i] = half * z * (one + I.tanh(inner))


def build(context):
    erf_artifact = context.compile("sub_gelu_erf", sub_gelu_erf_kernel)
    tanh_artifact = context.compile("sub_gelu_tanh", sub_gelu_tanh_kernel)

    def wrapper(input, other, alpha=1, approximate='none', out=None):
        if approximate == 'tanh':
            if out is None:
                out = torch.empty_like(input)
            tanh_artifact(input, other, alpha, out)
        else:
            if out is None:
                out = torch.empty_like(input)
            erf_artifact(input, other, alpha, out)
        return out

    return wrapper

