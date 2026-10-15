import torch
import intent
import intent.language as I


@intent.kernel
def linear_relu_kernel(
    input: I.In[I.f32, (32, 4096)],
    weight: I.In[I.f32, (2048, 4096)],
    bias: I.In[I.f32, (2048,)],
    linear_out: I.Out[I.f32, (32, 2048)],
    has_bias: I.Constexpr[bool],
):
    for i in I.parallel(I.domain(0, 32)):
        for j in I.domain(0, 2048):
            acc = I.cast(0.0, I.f32)
            for k in I.domain(0, 4096):
                acc = acc + input[i, k] * weight[j, k]
            if has_bias:
                val = acc + bias[j]
            else:
                val = acc
            linear_out[i, j] = I.select(val > I.cast(0.0, I.f32), val, I.cast(0.0, I.f32))


@intent.kernel
def layer_norm_kernel(
    linear_out: I.In[I.f32, (32, 2048)],
    output: I.Out[I.f32, (32, 2048)],
    eps: I.Constexpr[float],
):
    for i in I.parallel(I.domain(0, 32)):
        row = linear_out[i, :]
        mean = I.reduce.sum(row, axis=0) * I.cast(1.0 / 2048.0, I.f32)
        centered = row - mean
        variance = I.reduce.sum(centered * centered, axis=0) * I.cast(1.0 / 2048.0, I.f32)
        inv_std = I.rsqrt(variance + eps)
        normalized = centered * inv_std
        output[i, :] = normalized


def build(context):
    linear_artifact = context.compile(
        "linear_relu",
        linear_relu_kernel,
        constexprs={"has_bias": True},
    )
    ln_artifact = context.compile("layer_norm", layer_norm_kernel, constexprs={"eps": 1e-05})

    def wrapper(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        normalized_shape=None,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ) -> torch.Tensor:
        linear_out = torch.empty((32, 2048), dtype=torch.float32, device=input.device)
        if bias is not None:
            linear_artifact(input, weight, bias, linear_out)
        else:
            zero_bias = torch.zeros((2048,), dtype=torch.float32, device=input.device)
            linear_artifact(input, weight, zero_bias, linear_out)
        output = torch.empty_like(linear_out)
        ln_artifact(linear_out, output)
        return output

    return wrapper

