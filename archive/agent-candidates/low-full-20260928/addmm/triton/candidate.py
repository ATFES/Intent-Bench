import torch
import triton
import triton.language as tl


@triton.jit
def _addmm_kernel(
    input_ptr,
    mat1_ptr,
    mat2_ptr,
    out_ptr,
    M,
    N,
    K,
    alpha,
    beta,
    BETA_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(mat1_ptr + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(mat2_ptr + ks[:, None] * N + cols[None, :],
                    mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    if BETA_ZERO:
        result = acc * alpha
    else:
        addend = tl.load(input_ptr + rows[:, None] * N + cols[None, :],
                         mask=(rows[:, None] < M) & (cols[None, :] < N), other=0.0)
        result = acc * alpha + addend.to(tl.float32) * beta

    tl.store(out_ptr + rows[:, None] * N + cols[None, :], result,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def build(context):
    def wrapper(input, mat1, mat2, beta=1, alpha=1, out=None):
        M = mat1.shape[0]
        K = mat1.shape[1]
        N = mat2.shape[1]
        result = out if out is not None else torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        _addmm_kernel[grid](
            input, mat1, mat2, result, M, N, K, alpha, beta,
            BETA_ZERO=(beta == 0), BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
            num_warps=8, num_stages=4,
        )
        return result

    return wrapper
