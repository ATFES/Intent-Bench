import torch
import triton
import triton.language as tl


@triton.jit
def _linear_relu(x_ptr, w_ptr, b_ptr, y_ptr, m, k, n,
                 stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                 HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(n, BLOCK_N)
    pid_n = pid % tl.cdiv(n, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, k, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=(offs_m[:, None] < m) & (offs_k[None, :] < k), other=0.0)
        wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                     mask=(offs_n[:, None] < n) & (offs_k[None, :] < k), other=0.0)
        acc += tl.dot(xv, tl.trans(wv))
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n[None, :], mask=offs_n[None, :] < n, other=0.0)
    acc = tl.maximum(acc, 0.0)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def _layer_norm(x_ptr, y_ptr, rows, cols, stride_xm, stride_xn, stride_ym, stride_yn,
                eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / cols
    out = centered * tl.rsqrt(var + eps)
    tl.store(y_ptr + row * stride_ym + offs * stride_yn, out, mask=mask)


def build(context):
    def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None,
                                    eps=1e-5, elementwise_affine=True):
        rows = input.numel() // input.shape[-1]
        k = input.shape[-1]
        n = weight.shape[0]
        tmp = torch.empty((rows, n), device=input.device, dtype=torch.float32)
        out = torch.empty_like(tmp)
        block_m = 1
        block_n = 128
        block_k = 32
        _linear_relu[(rows * triton.cdiv(n, block_n),)](
            input, weight, bias, tmp, rows, k, n,
            input.stride(-2), input.stride(-1), weight.stride(0), weight.stride(1),
            tmp.stride(0), tmp.stride(1),
            HAS_BIAS=bias is not None, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
        _layer_norm[(rows,)](tmp, out, rows, n, tmp.stride(0), tmp.stride(1),
                      out.stride(0), out.stride(1), eps, BLOCK=2048)
        return out.reshape((*input.shape[:-1], n))

    return fused_layer_norm_relu_linear
