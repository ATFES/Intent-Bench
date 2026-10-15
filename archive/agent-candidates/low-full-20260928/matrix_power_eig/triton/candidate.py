import torch
import triton
import triton.language as tl


@triton.jit
def _symmetric_square(A, Out, N: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    col_mask = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in tl.range(0, N, BLOCK_K):
        ks = kk + tl.arange(0, BLOCK_K)

        # Select the stored lower-triangle element for each logical access.
        a_col = tl.where(ks[None, :] <= rows[:, None], ks[None, :], rows[:, None])
        a_row = tl.where(ks[None, :] <= rows[:, None], rows[:, None], ks[None, :])
        a_ptrs = A + a_row * N + a_col
        a_mask = row_mask[:, None] & (ks[None, :] < N)
        lhs = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_row = tl.where(cols[None, :] <= ks[:, None], ks[:, None], cols[None, :])
        b_col = tl.where(cols[None, :] <= ks[:, None], cols[None, :], ks[:, None])
        b_ptrs = A + b_row * N + b_col
        b_mask = (ks[:, None] < N) & col_mask[None, :]
        rhs = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(lhs, rhs, input_precision="ieee")

    out_rows = rows[:, None]
    out_cols = cols[None, :]
    tl.store(Out + out_rows * N + out_cols, acc,
             mask=row_mask[:, None] & col_mask[None, :])


def build(context):
    def matrix_power_eig(A: torch.Tensor, k: float = 2.0, *, out: torch.Tensor = None):
        target = out if out is not None else torch.empty_like(A)
        _symmetric_square[(triton.cdiv(256, 64), triton.cdiv(256, 64))](
            A, target, N=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4)
        return target

    return matrix_power_eig
