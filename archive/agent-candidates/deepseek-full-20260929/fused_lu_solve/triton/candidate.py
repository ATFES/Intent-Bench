import torch
import triton
import triton.language as tl


@triton.jit
def fused_lu_solve_kernel(
    A, b, LU, x,
    N,
    BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    valid_r = rows < N
    valid_c = cols < N
    zero_vec = tl.zeros((BLOCK_N,), tl.float32)
    zero_tile = tl.zeros((TILE_M,), tl.float32)
    neg_one_vec = tl.full((BLOCK_N,), -1.0, tl.float32)

    # Copy A into the scratch factorization buffer.  The input A is not modified.
    for r0 in tl.range(0, N, TILE_M):
        r_tile = r0 + tl.arange(0, TILE_M)
        rmask = r_tile < N
        for c0 in tl.range(0, N, TILE_N):
            c_tile = c0 + tl.arange(0, TILE_N)
            cmask = c_tile < N
            mask = rmask[:, None] & cmask[None, :]
            offs = r_tile[:, None] * N + c_tile[None, :]
            vals = tl.load(A + offs, mask=mask, other=0.0)
            tl.store(LU + offs, vals, mask=mask)
    tl.debug_barrier()

    bvec = tl.load(b + rows, mask=valid_r, other=0.0)

    # In-place partial-pivoting LU: the pivot row permutation is applied to
    # A only.  b is deliberately left unpermuted, as required by the task.
    for k in tl.range(0, N):
        col = tl.load(LU + rows * N + k, mask=valid_r, other=0.0)
        abs_col = tl.abs(col)
        candidates = tl.where(valid_r & (rows >= k), abs_col, neg_one_vec)
        p = tl.argmax(candidates, axis=0, tie_break_left=True)

        row_k = tl.load(LU + k * N + cols, mask=valid_c, other=0.0)
        row_p = tl.load(LU + p * N + cols, mask=valid_c, other=0.0)
        tl.store(LU + k * N + cols, row_p, mask=valid_c)
        tl.store(LU + p * N + cols, row_k, mask=valid_c)
        tl.debug_barrier()

        u_row = tl.load(LU + k * N + cols, mask=valid_c, other=0.0)
        piv = tl.sum(tl.where(cols == k, u_row, zero_vec), axis=0)

        # Rank-one update of the trailing submatrix, tiled to limit register
        # pressure, then store the multiplier column L[:, k].
        for r0 in tl.range(k + 1, N, TILE_M):
            r_tile = r0 + tl.arange(0, TILE_M)
            rmask = r_tile < N
            col_tile = tl.load(LU + r_tile * N + k, mask=rmask, other=0.0)
            mult_tile = tl.where(rmask & (r_tile > k), col_tile / piv, zero_tile)

            for c0 in tl.range(k + 1, N, TILE_N):
                c_tile = c0 + tl.arange(0, TILE_N)
                cmask = c_tile < N
                mask = rmask[:, None] & cmask[None, :]
                offs = r_tile[:, None] * N + c_tile[None, :]
                vals = tl.load(LU + offs, mask=mask, other=0.0)
                u_tile = tl.load(LU + k * N + c_tile, mask=cmask, other=0.0)
                updated = vals - mult_tile[:, None] * u_tile[None, :]
                tl.store(LU + offs, updated, mask=mask)

            tl.store(LU + r_tile * N + k, mult_tile, mask=rmask)
        tl.debug_barrier()

    # Solve L y = b with a unit lower triangular L.
    y = tl.zeros((BLOCK_N,), tl.float32)
    for i in tl.range(0, N):
        row = tl.load(LU + i * N + cols, mask=valid_c, other=0.0)
        partial = tl.where(cols < i, row * y, zero_vec)
        s = tl.sum(partial, axis=0)
        bi = tl.sum(tl.where(rows == i, bvec, zero_vec), axis=0)
        yi = bi - s
        y = tl.where(rows == i, yi, y)

    # Solve U x = y.
    xvec = tl.zeros((BLOCK_N,), tl.float32)
    for step in tl.range(0, N):
        i = N - 1 - step
        row = tl.load(LU + i * N + cols, mask=valid_c, other=0.0)
        yi = tl.sum(tl.where(rows == i, y, zero_vec), axis=0)
        partial = tl.where(cols > i, row * xvec, zero_vec)
        s = tl.sum(partial, axis=0)
        diag = tl.sum(tl.where(cols == i, row, zero_vec), axis=0)
        xi = (yi - s) / diag
        xvec = tl.where(rows == i, xi, xvec)

    tl.store(x + rows, xvec, mask=valid_r)


def build(context):
    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        N = A.shape[0]
        BLOCK_N = triton.next_power_of_2(N)
        LU = torch.empty_like(A)
        out = torch.empty_like(b)
        fused_lu_solve_kernel[(1,)](
            A, b, LU, out,
            N,
            BLOCK_N=BLOCK_N,
            TILE_M=64,
            TILE_N=64,
            num_warps=8,
        )
        return out

    return fused_lu_solve
