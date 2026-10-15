import torch
import triton
import triton.language as tl


@triton.jit
def _qr_solve_kernel(
    a_ptr,
    b_ptr,
    a_work_ptr,
    b_work_ptr,
    x_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < m

    for tile_start in range(0, n, BLOCK_N):
        cols = tile_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < n)
        values = tl.load(
            a_ptr + rows[:, None] * lda + cols[None, :],
            mask=mask,
            other=0.0,
        )
        tl.store(
            a_work_ptr + rows[:, None] * lda + cols[None, :],
            values,
            mask=mask,
        )

    for tile_start in range(0, k, BLOCK_K):
        cols = tile_start + tl.arange(0, BLOCK_K)
        mask = row_mask[:, None] & (cols[None, :] < k)
        values = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=mask,
            other=0.0,
        )
        tl.store(
            b_work_ptr + rows[:, None] * ldb + cols[None, :],
            values,
            mask=mask,
        )

    for col in range(0, n):
        column = tl.load(
            a_work_ptr + rows * lda + col,
            mask=row_mask,
            other=0.0,
        )
        scale = tl.max(tl.abs(column), axis=0)
        if scale > 0.0:
            scaled = column / scale
            norm = scale * tl.sqrt_rn(tl.sum(scaled * scaled, axis=0))
            diagonal = tl.load(a_work_ptr + col * lda + col)
            sign = tl.where(diagonal >= 0.0, 1.0, -1.0)
            alpha = -sign * norm
            reflector = tl.where(rows == col, diagonal - alpha, column)
            reflector_norm = tl.sum(reflector * reflector, axis=0)

            if reflector_norm > 0.0:
                tau = tl.div_rn(2.0, reflector_norm)

                for tile_start in range(col + 1, n, BLOCK_N):
                    cols = tile_start + tl.arange(0, BLOCK_N)
                    cols_mask = cols < n
                    update_mask = row_mask[:, None] & cols_mask[None, :]
                    block = tl.load(
                        a_work_ptr + rows[:, None] * lda + cols[None, :],
                        mask=update_mask,
                        other=0.0,
                    )
                    weights = tl.sum(reflector[:, None] * block, axis=0)
                    block = block - (tau * reflector)[:, None] * weights[None, :]
                    tl.store(
                        a_work_ptr + rows[:, None] * lda + cols[None, :],
                        block,
                        mask=update_mask,
                    )

                for tile_start in range(0, k, BLOCK_K):
                    cols = tile_start + tl.arange(0, BLOCK_K)
                    cols_mask = cols < k
                    update_mask = row_mask[:, None] & cols_mask[None, :]
                    block = tl.load(
                        b_work_ptr + rows[:, None] * ldb + cols[None, :],
                        mask=update_mask,
                        other=0.0,
                    )
                    weights = tl.sum(reflector[:, None] * block, axis=0)
                    block = block - (tau * reflector)[:, None] * weights[None, :]
                    tl.store(
                        b_work_ptr + rows[:, None] * ldb + cols[None, :],
                        block,
                        mask=update_mask,
                    )

    solution_cols = tl.arange(0, BLOCK_R)
    rhs_cols = tl.arange(0, BLOCK_K)
    rhs_mask = rhs_cols < k
    for step in range(0, n):
        row = n - 1 - step
        coefficient_mask = (solution_cols > row) & (solution_cols < n)
        coefficients = tl.load(
            a_work_ptr + row * lda + solution_cols,
            mask=coefficient_mask,
            other=0.0,
        )
        later_values = tl.load(
            x_ptr + solution_cols[:, None] * k + rhs_cols[None, :],
            mask=coefficient_mask[:, None] & rhs_mask[None, :],
            other=0.0,
        )
        accumulated = tl.sum(coefficients[:, None] * later_values, axis=0)
        rhs = tl.load(b_work_ptr + row * ldb + rhs_cols, mask=rhs_mask, other=0.0)
        diagonal = tl.load(a_work_ptr + row * lda + row)
        solution = tl.div_rn(rhs - accumulated, diagonal)
        tl.store(x_ptr + row * k + rhs_cols, solution, mask=rhs_mask)


def build(context):
    del context

    def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or b.ndim != 2:
            raise ValueError("A and b must both be two-dimensional")
        if A.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("A and b must have float32 dtype")
        if not A.is_cuda or not b.is_cuda:
            raise ValueError("A and b must be CUDA tensors")
        if A.device != b.device:
            raise ValueError("A and b must be on the same CUDA device")
        if A.shape[0] != b.shape[0] or A.shape[0] < A.shape[1]:
            raise ValueError("A must have shape (m, n) with m >= n and b shape (m, k)")

        m, n = A.shape
        k = b.shape[1]
        a_work = torch.empty_like(A)
        b_work = torch.empty_like(b)
        x = torch.empty((n, k), device=A.device, dtype=A.dtype)

        _qr_solve_kernel[(1,)](
            A,
            b,
            a_work,
            b_work,
            x,
            m,
            n,
            k,
            A.stride(0),
            b.stride(0),
            BLOCK_M=triton.next_power_of_2(m),
            BLOCK_N=32,
            BLOCK_K=16,
            BLOCK_R=triton.next_power_of_2(n),
            num_warps=16,
        )
        return x

    return fused_qr_solve
