import torch
import intent
import intent.language as I


@intent.kernel
def batch_norm_affine_kernel(
    input: I.In[I.f16, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    output: I.Out[I.f16, ("N", "C", "H", "W")],
    eps: I.f32,
):
    n_size, c_size, h_size, w_size = input.shape
    n = I.domain(0, n_size)
    c = I.domain(0, c_size)
    h = I.domain(0, h_size)
    w = I.domain(0, w_size)

    x = I.cast(input[n, c, h, w], I.f32)
    mean = I.reshape(running_mean[c], (1, c_size, 1, 1))
    variance = I.reshape(running_var[c], (1, c_size, 1, 1))
    affine_weight = I.reshape(weight[c], (1, c_size, 1, 1))
    affine_bias = I.reshape(bias[c], (1, c_size, 1, 1))

    normalized = (x - mean) * I.rsqrt(variance + eps)
    output[n, c, h, w] = I.cast(normalized * affine_weight + affine_bias, I.f16)


@intent.kernel
def batch_norm_no_affine_kernel(
    input: I.In[I.f16, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    output: I.Out[I.f16, ("N", "C", "H", "W")],
    eps: I.f32,
):
    n_size, c_size, h_size, w_size = input.shape
    n = I.domain(0, n_size)
    c = I.domain(0, c_size)
    h = I.domain(0, h_size)
    w = I.domain(0, w_size)

    x = I.cast(input[n, c, h, w], I.f32)
    mean = I.reshape(running_mean[c], (1, c_size, 1, 1))
    variance = I.reshape(running_var[c], (1, c_size, 1, 1))
    normalized = (x - mean) * I.rsqrt(variance + eps)
    output[n, c, h, w] = I.cast(normalized, I.f16)


@intent.kernel
def batch_norm_weight_only_kernel(
    input: I.In[I.f16, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    output: I.Out[I.f16, ("N", "C", "H", "W")],
    eps: I.f32,
):
    n_size, c_size, h_size, w_size = input.shape
    n = I.domain(0, n_size)
    c = I.domain(0, c_size)
    h = I.domain(0, h_size)
    w = I.domain(0, w_size)

    x = I.cast(input[n, c, h, w], I.f32)
    mean = I.reshape(running_mean[c], (1, c_size, 1, 1))
    variance = I.reshape(running_var[c], (1, c_size, 1, 1))
    affine_weight = I.reshape(weight[c], (1, c_size, 1, 1))
    normalized = (x - mean) * I.rsqrt(variance + eps)
    output[n, c, h, w] = I.cast(normalized * affine_weight, I.f16)


@intent.kernel
def batch_norm_bias_only_kernel(
    input: I.In[I.f16, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    output: I.Out[I.f16, ("N", "C", "H", "W")],
    eps: I.f32,
):
    n_size, c_size, h_size, w_size = input.shape
    n = I.domain(0, n_size)
    c = I.domain(0, c_size)
    h = I.domain(0, h_size)
    w = I.domain(0, w_size)

    x = I.cast(input[n, c, h, w], I.f32)
    mean = I.reshape(running_mean[c], (1, c_size, 1, 1))
    variance = I.reshape(running_var[c], (1, c_size, 1, 1))
    affine_bias = I.reshape(bias[c], (1, c_size, 1, 1))
    normalized = (x - mean) * I.rsqrt(variance + eps)
    output[n, c, h, w] = I.cast(normalized + affine_bias, I.f16)


def build(context):
    affine_artifact = context.compile("batch_norm_affine", batch_norm_affine_kernel)
    no_affine_artifact = context.compile("batch_norm_no_affine", batch_norm_no_affine_kernel)
    weight_only_artifact = context.compile("batch_norm_weight_only", batch_norm_weight_only_kernel)
    bias_only_artifact = context.compile("batch_norm_bias_only", batch_norm_bias_only_kernel)

    def batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        del training, momentum
        output = torch.empty_like(input)
        if weight is None and bias is None:
            no_affine_artifact(input, running_mean, running_var, output, eps)
        elif bias is None:
            weight_only_artifact(input, running_mean, running_var, weight, output, eps)
        elif weight is None:
            bias_only_artifact(input, running_mean, running_var, bias, output, eps)
        else:
            affine_artifact(input, running_mean, running_var, weight, bias, output, eps)
        return output

    return batch_norm
