import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < out_features
    k = tl.arange(0, in_features)

    x = tl.load(x_ptr + row * in_features + k)
    w = tl.load(w_ptr + cols[:, None] * in_features + k[None, :], mask=valid[:, None], other=0.0)
    scores = tl.dot(x[None, :], tl.trans(w), out_dtype=tl.float32)
    scores = scores[0, :]
    if has_bias:
        scores += tl.load(b_ptr + cols, mask=valid, other=0.0)

    row_max = tl.max(tl.where(valid, scores, -float("inf"),), axis=0)
    exp_scores = tl.exp(scores - row_max)
    denom = tl.sum(tl.where(valid, exp_scores, 0.0), axis=0)
    result = scores - row_max - tl.log(denom)
    tl.store(out_ptr + row * out_features + cols, result, mask=valid)


def build(context):
    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        out = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=input.dtype)
        n_rows = input.shape[0]
        out_features = weight.shape[0]
        block = triton.next_power_of_2(out_features)
        bias_arg = bias if bias is not None else input
        _log_softmax_linear_kernel[(n_rows,)](
            input,
            weight,
            bias_arg,
            out,
            n_rows,
            in_features=input.shape[1],
            out_features=out_features,
            has_bias=bias is not None,
            BLOCK=block,
        )
        return out

    return log_softmax_linear
