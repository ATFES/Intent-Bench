import torch
import triton
import triton.language as tl


@triton.jit
def _linear_logsoftmax(
    x_ptr, w_ptr, b_ptr, y_ptr,
    n_rows, in_features, out_features,
    has_bias: tl.constexpr,
    stride_x0, stride_x1, stride_w0, stride_w1, stride_y0, stride_y1,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid_n = cols < out_features
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # The fixed profile has contiguous rows and 256 features; blocking K keeps
    # the matrix product in dot instructions while retaining the fusion.
    for k0 in tl.range(0, in_features, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + row * stride_x0 + ks * stride_x1,
                    mask=ks < in_features, other=0.0)
        w = tl.load(w_ptr + cols[:, None] * stride_w0 + ks[None, :] * stride_w1,
                    mask=valid_n[:, None] & (ks[None, :] < in_features), other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
    if has_bias:
        acc += tl.load(b_ptr + cols, mask=valid_n, other=0.0)
    acc = tl.where(valid_n, acc, -float("inf"))
    m = tl.max(acc, axis=0)
    e = tl.exp(acc - m)
    denom = tl.sum(e, axis=0)
    out = acc - m - tl.log(denom)
    tl.store(y_ptr + row * stride_y0 + cols * stride_y1, out, mask=valid_n)


def build(context):
    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        out = torch.empty(input.shape[:-1] + (weight.shape[0],), device=input.device,
                          dtype=input.dtype)
        n_rows = input.numel() // input.shape[-1]
        _linear_logsoftmax[(n_rows,)](
            input, weight, bias if bias is not None else input, out,
            n_rows, input.shape[-1], weight.shape[0],
            has_bias=bias is not None,
            stride_x0=input.stride(-2) if input.ndim > 1 else input.shape[-1],
            stride_x1=input.stride(-1),
            stride_w0=weight.stride(0), stride_w1=weight.stride(1),
            stride_y0=out.stride(-2) if out.ndim > 1 else out.shape[-1],
            stride_y1=out.stride(-1),
            BLOCK_N=256, BLOCK_K=32,
        )
        return out
    return log_softmax_linear
