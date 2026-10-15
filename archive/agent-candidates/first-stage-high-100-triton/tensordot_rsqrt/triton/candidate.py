import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _tensordot_rsqr_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Grouping neighboring M tiles improves L2 reuse of the B tiles.
    group_size = GROUP_M * grid_n
    group_id = pid // group_size
    first_m = group_id * GROUP_M
    group_m = tl.minimum(grid_m - first_m, GROUP_M)
    pid_in_group = pid % group_size
    pid_m = first_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
        b_mask = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    result = tl.rsqrt(acc)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        result,
        mask=out_mask,
    )


def build(context):
    def wrapper(a, b, dims):
        M = a.shape[0]
        K = a.shape[1]
        N = b.shape[1]
        output = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        _tensordot_rsqr_kernel[grid](
            a,
            b,
            output,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
        )
        return output

    return wrapper
