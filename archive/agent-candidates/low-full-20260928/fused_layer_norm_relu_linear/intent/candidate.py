import torch
import intent
import intent.language as I


@intent.kernel
def _linear_relu(
    x: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, weight.shape[0])
    product = I.matmul(x, weight, acc_dtype=I.f32, transpose_rhs=True)
    product = product + bias
    product = I.select(product > I.cast(0.0, I.f32), product, I.cast(0.0, I.f32))
    out[rows, cols] = product


@intent.kernel
def _linear_relu_no_bias(
    x: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    out: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, weight.shape[0])
    product = I.matmul(x, weight, acc_dtype=I.f32, transpose_rhs=True)
    product = I.select(product > I.cast(0.0, I.f32), product, I.cast(0.0, I.f32))
    out[rows, cols] = product


@intent.kernel
def _layer_norm(
    values: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, ("M", "N")],
    eps: I.f32,
):
    rows = I.domain(0, values.shape[0])
    cols = I.domain(0, values.shape[1])
    for row in I.parallel(rows):
        v = values[row, cols]
        mean = I.reduce.sum(v, axis=0, acc_dtype=I.f32) / I.cast(values.shape[1], I.f32)
        centered = v - mean
        variance = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32) / I.cast(values.shape[1], I.f32)
        normalized = centered / I.sqrt(variance + eps)
        out[row, cols] = normalized


def build(context):
    linear = context.compile("linear_relu", _linear_relu)
    linear_no_bias = context.compile("linear_relu_no_bias", _linear_relu_no_bias)
    normalize = context.compile("layer_norm", _layer_norm)

    def fused_layer_norm_relu_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        normalized_shape: torch.Size = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.Tensor:
        m = input.shape[0]
        n = weight.shape[0]
        intermediate = torch.empty((m, n), device=input.device, dtype=input.dtype)
        output = torch.empty((m, n), device=input.device, dtype=input.dtype)
        if bias is None:
            linear_no_bias(input, weight, intermediate)
        else:
            linear(input, weight, bias, intermediate)
        normalize(intermediate, output, eps)
        return output

    return fused_layer_norm_relu_linear
