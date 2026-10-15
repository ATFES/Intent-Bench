import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_m = 8
    group_size = group_m * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * group_m
    group_pid_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_pid_m)
    pid_n = (pid % group_size) // group_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        a_tile = tl.load(a, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b_tile = tl.load(b, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a_tile, b_tile)
        a += BLOCK_K * stride_ak
        b += BLOCK_K * stride_bk
    c = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def build(context):
    def wrapper(tensor1, tensor2):
        M, K = tensor1.shape
        _, N = tensor2.shape
        output = torch.empty((M, N), device=tensor1.device, dtype=tensor1.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        _matmul_kernel[grid](
            tensor1, tensor2, output,
            M, N, K,
            tensor1.stride(0), tensor1.stride(1),
            tensor2.stride(0), tensor2.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    return wrapper
