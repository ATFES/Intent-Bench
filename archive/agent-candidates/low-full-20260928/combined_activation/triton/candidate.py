import torch
import triton
import triton.language as tl


@triton.jit
def _combined_activation_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    d_in,
    d_out,
    stride_x0,
    stride_x1,
    stride_w10,
    stride_w11,
    stride_w20,
    stride_b0,
    stride_o0,
    stride_o1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, d_in, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * stride_x0 + ks[None, :] * stride_x1,
            mask=(rows[:, None] < n_rows) & (ks[None, :] < d_in),
            other=0.0,
        )
        w = tl.load(
            w1_ptr + ks[:, None] * stride_w10 + cols[None, :] * stride_w11,
            mask=(ks[:, None] < d_in) & (cols[None, :] < d_out),
            other=0.0,
        )
        acc += tl.dot(x, w)

    z = 1.0 / (1.0 + tl.exp(-acc))
    activated = tl.math.tanh(z)
    w2 = tl.load(w2_ptr + cols * stride_w20, mask=cols < d_out, other=0.0)
    bias = tl.load(bias_ptr + cols * stride_b0, mask=cols < d_out, other=0.0)
    result = activated * w2[None, :] + bias[None, :]
    mask = (rows[:, None] < n_rows) & (cols[None, :] < d_out)
    tl.store(out_ptr + rows[:, None] * stride_o0 + cols[None, :] * stride_o1, result, mask=mask)


def build(context):
    def combined_activation(input, weight1, weight2, bias, *, out=None):
        n_rows = input.numel() // input.shape[-1]
        d_in = input.shape[-1]
        d_out = weight1.shape[-1]
        if out is None:
            output = torch.empty((*input.shape[:-1], d_out), device=input.device, dtype=input.dtype)
        else:
            output = out
        _combined_activation_kernel[(triton.cdiv(n_rows, 128), triton.cdiv(d_out, 128))](
            input,
            weight1,
            weight2,
            bias,
            output,
            n_rows,
            d_in,
            d_out,
            input.stride(-2),
            input.stride(-1),
            weight1.stride(0),
            weight1.stride(1),
            weight2.stride(-1),
            bias.stride(-1),
            output.stride(-2),
            output.stride(-1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=8,
            num_stages=4,
        )
        return output

    return combined_activation
