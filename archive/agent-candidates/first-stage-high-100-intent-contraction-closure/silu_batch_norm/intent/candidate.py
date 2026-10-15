import torch
import intent
import intent.language as I


@intent.kernel
def _silu_batch_norm_affine(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    eps: I.f32,
    output: I.Out[I.f32, ("N", "C", "H", "W")],
):
    n, c, h, w = input.shape
    batch = I.domain(0, n)
    channel = I.domain(0, c)
    row = I.domain(0, h)
    column = I.domain(0, w)

    value = input[batch, channel, row, column]
    mean = I.reshape(running_mean[channel], (1, c, 1, 1))
    variance = I.reshape(running_var[channel], (1, c, 1, 1))
    scale = I.reshape(weight[channel], (1, c, 1, 1))
    offset = I.reshape(bias[channel], (1, c, 1, 1))

    normalized = (value - mean) * I.rsqrt(variance + eps)
    normalized = normalized * scale + offset
    one = I.cast(1.0, I.f32)
    sigmoid = I.fdiv(one, one + I.exp(-normalized))
    output[batch, channel, row, column] = normalized * sigmoid


@intent.kernel
def _silu_batch_norm_weight_only(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    eps: I.f32,
    output: I.Out[I.f32, ("N", "C", "H", "W")],
):
    n, c, h, w = input.shape
    batch = I.domain(0, n)
    channel = I.domain(0, c)
    row = I.domain(0, h)
    column = I.domain(0, w)

    value = input[batch, channel, row, column]
    mean = I.reshape(running_mean[channel], (1, c, 1, 1))
    variance = I.reshape(running_var[channel], (1, c, 1, 1))
    scale = I.reshape(weight[channel], (1, c, 1, 1))

    normalized = (value - mean) * I.rsqrt(variance + eps)
    normalized = normalized * scale
    one = I.cast(1.0, I.f32)
    sigmoid = I.fdiv(one, one + I.exp(-normalized))
    output[batch, channel, row, column] = normalized * sigmoid


@intent.kernel
def _silu_batch_norm_bias_only(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    eps: I.f32,
    output: I.Out[I.f32, ("N", "C", "H", "W")],
):
    n, c, h, w = input.shape
    batch = I.domain(0, n)
    channel = I.domain(0, c)
    row = I.domain(0, h)
    column = I.domain(0, w)

    value = input[batch, channel, row, column]
    mean = I.reshape(running_mean[channel], (1, c, 1, 1))
    variance = I.reshape(running_var[channel], (1, c, 1, 1))
    offset = I.reshape(bias[channel], (1, c, 1, 1))

    normalized = (value - mean) * I.rsqrt(variance + eps)
    normalized = normalized + offset
    one = I.cast(1.0, I.f32)
    sigmoid = I.fdiv(one, one + I.exp(-normalized))
    output[batch, channel, row, column] = normalized * sigmoid


@intent.kernel
def _silu_batch_norm_no_affine(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    eps: I.f32,
    output: I.Out[I.f32, ("N", "C", "H", "W")],
):
    n, c, h, w = input.shape
    batch = I.domain(0, n)
    channel = I.domain(0, c)
    row = I.domain(0, h)
    column = I.domain(0, w)

    value = input[batch, channel, row, column]
    mean = I.reshape(running_mean[channel], (1, c, 1, 1))
    variance = I.reshape(running_var[channel], (1, c, 1, 1))

    normalized = (value - mean) * I.rsqrt(variance + eps)
    one = I.cast(1.0, I.f32)
    sigmoid = I.fdiv(one, one + I.exp(-normalized))
    output[batch, channel, row, column] = normalized * sigmoid


def build(context):
    affine = context.compile("silu_batch_norm_affine", _silu_batch_norm_affine, constexprs={})
    weight_only = context.compile(
        "silu_batch_norm_weight_only", _silu_batch_norm_weight_only, constexprs={}
    )
    bias_only = context.compile(
        "silu_batch_norm_bias_only", _silu_batch_norm_bias_only, constexprs={}
    )
    no_affine = context.compile(
        "silu_batch_norm_no_affine", _silu_batch_norm_no_affine, constexprs={}
    )

    def silu_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        output = torch.empty_like(input)
        if weight is None:
            if bias is None:
                no_affine(input, running_mean, running_var, eps, output)
            else:
                bias_only(input, running_mean, running_var, bias, eps, output)
        elif bias is None:
            weight_only(input, running_mean, running_var, weight, eps, output)
        else:
            affine(input, running_mean, running_var, weight, bias, eps, output)
        return output

    return silu_batch_norm
