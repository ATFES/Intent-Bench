import torch
import intent
import intent.language as I


@intent.kernel
def _linear_relu_bias(
    input: I.In[I.f32, (32, 4096)],
    weight: I.In[I.f32, (2048, 4096)],
    bias: I.In[I.f32, (2048,)],
    output: I.Out[I.f32, (32, 2048)],
):
    rows = I.domain(0, 32)
    cols = I.domain(0, 2048)
    x = I.matmul(
        input,
        weight,
        acc_dtype=I.f32,
        transpose_rhs=True,
    )
    y = I.maximum(x + bias, I.cast(0.0, I.f32))
    output[rows, cols] = y


@intent.kernel
def _linear_relu_no_bias(
    input: I.In[I.f32, (32, 4096)],
    weight: I.In[I.f32, (2048, 4096)],
    output: I.Out[I.f32, (32, 2048)],
):
    rows = I.domain(0, 32)
    cols = I.domain(0, 2048)
    x = I.matmul(
        input,
        weight,
        acc_dtype=I.f32,
        transpose_rhs=True,
    )
    output[rows, cols] = I.maximum(x, I.cast(0.0, I.f32))


@intent.kernel
def _layer_norm(
    input: I.In[I.f32, (32, 2048)],
    eps: I.f32,
    output: I.Out[I.f32, (32, 2048)],
):
    rows = I.domain(0, 32)
    cols = I.domain(0, 2048)
    values = input[rows, cols]
    mean = I.reduce.sum(values, axis=1, acc_dtype=I.f32) / I.cast(2048, I.f32)
    mean_2d = I.reshape(mean, (32, 1))
    centered = values - mean_2d
    variance = I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32) / I.cast(2048, I.f32)
    scale = I.reshape(I.sqrt(variance + eps), (32, 1))
    output[rows, cols] = centered / scale


def build(context):
    linear_bias = context.compile(
        "fused_linear_relu_bias", _linear_relu_bias, constexprs={}
    )
    linear_no_bias = context.compile(
        "fused_linear_relu_no_bias", _linear_relu_no_bias, constexprs={}
    )
    layer_norm = context.compile("fused_layer_norm", _layer_norm, constexprs={})

    def fused_layer_norm_relu_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        normalized_shape: torch.Size = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.Tensor:
        intermediate = torch.empty((32, 2048), device=input.device, dtype=input.dtype)
        if bias is None:
            linear_no_bias(input, weight, intermediate)
        else:
            linear_bias(input, weight, bias, intermediate)
        output = torch.empty((32, 2048), device=input.device, dtype=input.dtype)
        layer_norm(intermediate, eps, output)
        return output

    return fused_layer_norm_relu_linear
