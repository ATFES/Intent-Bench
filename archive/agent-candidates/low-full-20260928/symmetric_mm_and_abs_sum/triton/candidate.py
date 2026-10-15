import torch
import triton
import triton.language as tl


@triton.jit
def _zero_scalar(Out):
    if tl.program_id(0) == 0:
        tl.store(Out, 0.0)


@triton.jit
def _symmetric_mm_abs_sum(
    A, C, Out,
    n, m,
    alpha, beta,
    stride_am, stride_ak,
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n
    col_mask = cols < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, m, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + rows[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=row_mask[:, None] & (ks[None, :] < m),
            other=0.0,
        )
        # A.T[k, col] is A[col, k].
        b = tl.load(
            A + cols[None, :] * stride_am + ks[:, None] * stride_ak,
            mask=(ks[:, None] < m) & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    c = tl.load(
        C + rows[:, None] * stride_cm + cols[None, :] * stride_ck,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    value = alpha * acc + beta * c
    contribution = tl.sum(tl.sum(tl.abs(value), axis=0), axis=0)
    tl.atomic_add(Out, contribution)


def build(context):
    def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor,
                                 alpha: float = 1.0,
                                 beta: float = 0.5) -> torch.Tensor:
        n = A.shape[0]
        m = A.shape[1]
        out = torch.empty((), device=A.device, dtype=torch.float32)
        _zero_scalar[(1,)](out)
        grid = (triton.cdiv(n, 64) * triton.cdiv(n, 64),)
        _symmetric_mm_abs_sum[grid](
            A, C, out, n, m, alpha, beta,
            A.stride(0), A.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
        return out

    return symmetric_mm_and_abs_sum
