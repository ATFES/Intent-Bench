import torch
import intent
from intent import language as I


@intent.kernel
def _softplus_linear_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    z = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    z = z + bias[None, :]
    scaled = beta * z
    softplus = I.fdiv(I.log1p(I.exp(scaled)), beta)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = I.select(z > threshold, z, softplus)


@intent.kernel
def _softplus_linear_no_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    z = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    scaled = beta * z
    softplus = I.fdiv(I.log1p(I.exp(scaled)), beta)
    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = I.select(z > threshold, z, softplus)


def build(context):
    bias_artifact = context.compile("softplus_linear_bias", _softplus_linear_bias)
    no_bias_artifact = context.compile("softplus_linear_no_bias", _softplus_linear_no_bias)

    def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
        output = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=input.dtype,
        )
        if bias is None:
            no_bias_artifact(input, weight, output, beta, threshold)
        else:
            bias_artifact(input, weight, bias, output, beta, threshold)
        return output

    return softplus_linear
