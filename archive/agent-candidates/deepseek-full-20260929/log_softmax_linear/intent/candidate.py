import torch

import intent
import intent.language as I


@intent.kernel
def _zero_bias(bias: I.Out[I.f32, ("N",)]):
    cols = I.domain(0, bias.shape[0])
    bias[cols] = I.cast(0.0, I.f32)


@intent.kernel
def _log_softmax_linear_rank2(
    input: I.In[I.f32, ("B", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("B", "N")],
    AXIS: I.Constexpr[int],
):
    B = input.shape[0]
    K = input.shape[1]
    N = weight.shape[0]

    rows = I.domain(0, B)
    in_cols = I.domain(0, K)
    out_cols = I.domain(0, N)

    lin = I.matmul(
        input[rows, in_cols],
        weight[out_cols, in_cols],
        acc_dtype=I.f32,
        transpose_rhs=True,
    ) + bias[out_cols]

    if AXIS == 1:
        row_max = I.reduce.max(lin, axis=1, acc_dtype=I.f32)
        shifted = lin - I.reshape(row_max, (B, 1))
        exp_vals = I.exp(shifted)
        row_sum = I.reduce.sum(exp_vals, axis=1, acc_dtype=I.f32)
        out[rows, out_cols] = shifted - I.reshape(I.log(row_sum), (B, 1))
    else:
        col_max = I.reduce.max(lin, axis=0, acc_dtype=I.f32)
        shifted = lin - I.reshape(col_max, (1, N))
        exp_vals = I.exp(shifted)
        col_sum = I.reduce.sum(exp_vals, axis=0, acc_dtype=I.f32)
        out[rows, out_cols] = shifted - I.reshape(I.log(col_sum), (1, N))


def build(context):
    axis0 = context.compile(
        "log_softmax_linear_axis0",
        _log_softmax_linear_rank2,
        constexprs={"AXIS": 0},
    )
    axis1 = context.compile(
        "log_softmax_linear_axis1",
        _log_softmax_linear_rank2,
        constexprs={"AXIS": 1},
    )
    zero_bias = context.compile("log_softmax_linear_zero_bias", _zero_bias)

    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if dtype is not None and dtype != torch.float32:
            raise NotImplementedError("only float32 is supported for this profile")
        if input.dim() != 2 or weight.dim() != 2:
            raise NotImplementedError("only rank-2 inputs are supported for this profile")

        out_features = weight.shape[0]
        out_shape = input.shape[:-1] + (out_features,)
        out = torch.empty(out_shape, device=input.device, dtype=torch.float32)

        if bias is None:
            bias = torch.empty((out_features,), device=input.device, dtype=torch.float32)
            zero_bias(bias)

        axis = dim if dim >= 0 else dim + input.dim()
        if axis == 1:
            axis1(input, weight, bias, out)
        elif axis == 0:
            axis0(input, weight, bias, out)
        else:
            raise NotImplementedError("only log_softmax over rank-2 axis 0 or -1 is supported")

        return out

    return log_softmax_linear
