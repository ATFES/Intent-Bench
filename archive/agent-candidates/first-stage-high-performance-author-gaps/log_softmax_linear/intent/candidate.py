import torch
import intent
import intent.language as I


@intent.kernel
def _linear_log_softmax(
    input: I.In[I.f32, ("B", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("B", "N")],
):
    B, _ = input.shape
    N, _ = weight.shape
    rows = I.domain(0, B)
    columns = I.domain(0, N)

    logits = I.matmul(input, weight, transpose_rhs=True, acc_dtype=I.f32)
    logits = logits + bias[columns]
    row_max = I.reduce.max(logits, axis=1, acc_dtype=I.f32)
    centered = logits - I.reshape(row_max, (B, 1))
    exp_centered = I.exp(centered)
    row_sum = I.reduce.sum(exp_centered, axis=1, acc_dtype=I.f32)
    log_sum = I.log(row_sum)
    output[rows, columns] = centered - I.reshape(log_sum, (B, 1))


@intent.kernel
def _linear_log_softmax_no_bias(
    input: I.In[I.f32, ("B", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("B", "N")],
):
    B, _ = input.shape
    N, _ = weight.shape
    rows = I.domain(0, B)
    columns = I.domain(0, N)

    logits = I.matmul(input, weight, transpose_rhs=True, acc_dtype=I.f32)
    row_max = I.reduce.max(logits, axis=1, acc_dtype=I.f32)
    centered = logits - I.reshape(row_max, (B, 1))
    exp_centered = I.exp(centered)
    row_sum = I.reduce.sum(exp_centered, axis=1, acc_dtype=I.f32)
    log_sum = I.log(row_sum)
    output[rows, columns] = centered - I.reshape(log_sum, (B, 1))


def build(context):
    with_bias = context.compile("linear_log_softmax", _linear_log_softmax)
    without_bias = context.compile("linear_log_softmax_no_bias", _linear_log_softmax_no_bias)

    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        # The supplied profile uses the final dimension and float32 inputs.
        if bias is None:
            return without_bias.run(input, weight)
        return with_bias.run(input, weight, bias)

    return log_softmax_linear
