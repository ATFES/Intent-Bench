import torch
import triton
import triton.language as tl


@triton.jit
def _tensordot_rsqrt_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < m) & (k0 + offs_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k0 + offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a, b)

    result = tl.rsqrt(acc)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        result,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def build(context):
    def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims):
        if dims != 1:
            raise ValueError("this fixed-profile implementation requires dims=1")
        m, k = a.shape
        kb, n = b.shape
        if k != kb:
            raise ValueError("contracted dimensions must match")
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(m, 128) * triton.cdiv(n, 128),)
        _tensordot_rsqrt_kernel[grid](
            a,
            b,
            out,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=8,
            num_stages=4,
        )
        return out

    return tensordot_rsqrt
