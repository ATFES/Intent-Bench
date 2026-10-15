from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def lu_factor_kernel(
    A: I.In[I.f32, ("N", "N")],
    LU: I.Out[I.f32, ("N", "N")],
    pivots: I.Out[I.i64, ("N",)],
    PIVOT: I.Constexpr[bool],
):
    N = A.shape[0]
    rows = I.domain(0, N)
    cols = I.domain(0, N)

    LU[rows, cols] = A[rows, cols]

    for col in I.domain(0, N):
        if PIVOT:
            tail_from_col = rows[col : N]
            cand = LU[tail_from_col, col]
            maxval, rel = I.arg_reduce.max(I.abs(cand), axis=0)
            pivot = col + rel

            pivots[col] = I.cast(pivot, I.i64)

            row_col = LU[col, cols]
            row_pivot = LU[pivot, cols]
            LU[col, cols] = row_pivot
            LU[pivot, cols] = row_col
        else:
            pivots[col] = I.cast(col, I.i64)

        if col + 1 < N:
            tail = rows[col + 1 : N]
            right = cols[col + 1 : N]

            diag = LU[col, col]
            lcol = LU[tail, col] / diag
            LU[tail, col] = lcol

            urow = LU[col, right]
            LU[tail, right] = LU[tail, right] - I.outer(lcol, urow)


@intent.kernel
def lu_solve_kernel(
    LU: I.In[I.f32, ("N", "N")],
    pivots: I.In[I.i64, ("N",)],
    B: I.In[I.f32, ("N", "K")],
    work: I.InOut[I.f32, ("N", "K")],
    X: I.Out[I.f32, ("N", "K")],
    PIVOT: I.Constexpr[bool],
):
    N = B.shape[0]
    K = B.shape[1]
    rows = I.domain(0, N)
    cols = I.domain(0, K)

    work[rows, cols] = B[rows, cols]

    if PIVOT:
        for col in I.domain(0, N):
            p = I.cast(pivots[col], I.index)
            row_col = work[col, cols]
            row_pivot = work[p, cols]
            work[col, cols] = row_pivot
            work[p, cols] = row_col

    # Forward substitution: L y = (P^T) b.  L has a unit diagonal.
    for j in I.domain(0, N):
        if j + 1 < N:
            below = rows[j + 1 : N]
            yrow = work[j, cols]
            lcol = LU[below, j]
            work[below, cols] = work[below, cols] - I.outer(lcol, yrow)

    # Backward substitution: U x = y.
    for ord in I.domain(0, N):
        j = N - 1 - ord
        ujj = LU[j, j]
        xrow = work[j, cols] / ujj
        X[j, cols] = xrow

        if j > 0:
            above = rows[0:j]
            ucol = LU[above, j]
            work[above, cols] = work[above, cols] - I.outer(ucol, xrow)


def _solve_rank2(A, Bs, pivot, out, lu_art, solve_art):
    device = A.device
    n = A.shape[-1]
    k = Bs.shape[-1]

    LU = torch.empty((n, n), dtype=torch.float32, device=device)
    pivots = torch.empty((n,), dtype=torch.int64, device=device)
    work = torch.empty((n, k), dtype=torch.float32, device=device)
    X = out if out is not None else torch.empty((n, k), dtype=torch.float32, device=device)

    lu_art(A, LU, pivots)
    solve_art(LU, pivots, Bs, work, X)
    return X


def _unravel_batch_index(flat, batch_shape):
    if not batch_shape:
        return ()
    idx = [0] * len(batch_shape)
    rem = flat
    for axis in range(len(batch_shape) - 1, -1, -1):
        extent = batch_shape[axis]
        idx[axis] = rem % extent
        rem //= extent
    return tuple(idx)


def build(context):
    lu_pivot = context.compile(
        "solve_multiple_lu_factor_pivot", lu_factor_kernel, constexprs={"PIVOT": True}
    )
    lu_nopivot = context.compile(
        "solve_multiple_lu_factor_nopivot", lu_factor_kernel, constexprs={"PIVOT": False}
    )
    solve_pivot = context.compile(
        "solve_multiple_lu_solve_pivot", lu_solve_kernel, constexprs={"PIVOT": True}
    )
    solve_nopivot = context.compile(
        "solve_multiple_lu_solve_nopivot", lu_solve_kernel, constexprs={"PIVOT": False}
    )

    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        lu_art = lu_pivot if pivot else lu_nopivot
        solve_art = solve_pivot if pivot else solve_nopivot

        if A.dim() == 2:
            X = _solve_rank2(A, Bs, pivot, out, lu_art, solve_art)
            return out if out is not None else X

        batch_shape = tuple(A.shape[:-2])
        total = 1
        for extent in batch_shape:
            total *= extent

        Xs = out if out is not None else torch.empty_like(Bs)
        for flat in range(total):
            batch_index = _unravel_batch_index(flat, batch_shape)
            _solve_rank2(A[batch_index], Bs[batch_index], pivot, Xs[batch_index], lu_art, solve_art)

        return out if out is not None else Xs

    return solve_multiple_lu
