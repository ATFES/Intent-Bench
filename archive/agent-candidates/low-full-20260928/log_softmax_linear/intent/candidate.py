import torch
import intent
import intent.language as I


@intent.kernel
def _linear_log_softmax(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    # Keep the GEMM and the normalization in one launch.  The row maximum is
    # materialized as a size-one axis so that all later expressions broadcast.
    logits = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    logits = logits + bias
    row_max = I.reduce.max(logits, axis=1, acc_dtype=I.f32)
    centered = logits - I.reshape(row_max, (32, 1))
    exponent = I.exp(centered)
    row_sum = I.reduce.sum(exponent, axis=1, acc_dtype=I.f32)
    result = centered - I.reshape(I.log(row_sum), (32, 1))
    output[:, :] = result


@intent.kernel
def _linear_log_softmax_no_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
):
    logits = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    row_max = I.reduce.max(logits, axis=1, acc_dtype=I.f32)
    centered = logits - I.reshape(row_max, (32, 1))
    exponent = I.exp(centered)
    row_sum = I.reduce.sum(exponent, axis=1, acc_dtype=I.f32)
    result = centered - I.reshape(I.log(row_sum), (32, 1))
    output[:, :] = result


def build(context):
    with_bias = context.compile("linear_log_softmax", _linear_log_softmax)
    without_bias = context.compile("linear_log_softmax_no_bias", _linear_log_softmax_no_bias)

    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        m = input.shape[0]
        n = weight.shape[0]
        output = torch.empty((m, n), device=input.device, dtype=input.dtype)
        if bias is None:
            without_bias(input, weight, output)
        else:
            with_bias(input, weight, bias, output)
        return output

    return log_softmax_linear
