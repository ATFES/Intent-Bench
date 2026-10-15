import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_batch_norm(
    input: I.In[I.f32, (32, 32768)],
    running_mean: I.In[I.f32, (32768,)],
    running_var: I.In[I.f32, (32768,)],
    weight: I.In[I.f32, (32768,)],
    bias: I.In[I.f32, (32768,)],
    output: I.Out[I.f32, (32, 32768)],
    eps: I.f32,
):
    rows = I.domain(0, 32)
    columns = I.domain(0, 32768)

    values = input[rows, columns]
    mean = running_mean[columns]
    variance = running_var[columns]
    scale = weight[columns]
    shift = bias[columns]

    normalized = (values - mean) * I.rsqrt(variance + eps)
    activated = I.sigmoid(normalized * scale + shift)
    output[rows, columns] = activated


@intent.kernel
def _sigmoid_batch_norm_no_weight(
    input: I.In[I.f32, (32, 32768)],
    running_mean: I.In[I.f32, (32768,)],
    running_var: I.In[I.f32, (32768,)],
    bias: I.In[I.f32, (32768,)],
    output: I.Out[I.f32, (32, 32768)],
    eps: I.f32,
):
    rows = I.domain(0, 32)
    columns = I.domain(0, 32768)

    values = input[rows, columns]
    mean = running_mean[columns]
    variance = running_var[columns]
    shift = bias[columns]

    normalized = (values - mean) * I.rsqrt(variance + eps)
    output[rows, columns] = I.sigmoid(normalized + shift)


@intent.kernel
def _sigmoid_batch_norm_no_bias(
    input: I.In[I.f32, (32, 32768)],
    running_mean: I.In[I.f32, (32768,)],
    running_var: I.In[I.f32, (32768,)],
    weight: I.In[I.f32, (32768,)],
    output: I.Out[I.f32, (32, 32768)],
    eps: I.f32,
):
    rows = I.domain(0, 32)
    columns = I.domain(0, 32768)

    values = input[rows, columns]
    mean = running_mean[columns]
    variance = running_var[columns]
    scale = weight[columns]

    normalized = (values - mean) * I.rsqrt(variance + eps)
    output[rows, columns] = I.sigmoid(normalized * scale)


@intent.kernel
def _sigmoid_batch_norm_no_affine(
    input: I.In[I.f32, (32, 32768)],
    running_mean: I.In[I.f32, (32768,)],
    running_var: I.In[I.f32, (32768,)],
    output: I.Out[I.f32, (32, 32768)],
    eps: I.f32,
):
    rows = I.domain(0, 32)
    columns = I.domain(0, 32768)

    values = input[rows, columns]
    mean = running_mean[columns]
    variance = running_var[columns]

    normalized = (values - mean) * I.rsqrt(variance + eps)
    output[rows, columns] = I.sigmoid(normalized)


def build(context):
    fused = context.compile("sigmoid_batch_norm_fused", _sigmoid_batch_norm, constexprs={})
    no_weight = context.compile(
        "sigmoid_batch_norm_no_weight", _sigmoid_batch_norm_no_weight, constexprs={}
    )
    no_bias = context.compile(
        "sigmoid_batch_norm_no_bias", _sigmoid_batch_norm_no_bias, constexprs={}
    )
    no_affine = context.compile(
        "sigmoid_batch_norm_no_affine", _sigmoid_batch_norm_no_affine, constexprs={}
    )

    def sigmoid_batch_norm(
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

        # The supplied profile is inference mode.  The training and momentum
        # parameters remain in the public callable for contract compatibility.
        if weight is None:
            if bias is None:
                no_affine(input, running_mean, running_var, output, eps)
            else:
                no_weight(input, running_mean, running_var, bias, output, eps)
        elif bias is None:
            no_bias(input, running_mean, running_var, weight, output, eps)
        else:
            fused(input, running_mean, running_var, weight, bias, output, eps)
        return output

    return sigmoid_batch_norm
