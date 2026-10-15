import torch
import intent
import intent.language as I


N = 256


@intent.kernel
def lu_factor(
    A: I.In[I.f32, (256, 256)],
    B: I.In[I.f32, (256, 1)],
    LU: I.Out[I.f32, (256, 257)],
):
    rows = I.domain(0, 256)
    cols = I.domain(0, 256)
    augmented_cols = I.domain(0, 257)

    # Materialize the matrix and RHS in one mutable logical state.
    for i in I.parallel(rows):
        for j in I.parallel(cols):
            LU[i, j] = A[i, j]
        LU[i, 256] = B[i, 0]

    # Doolittle LU with partial pivoting.  The outer loop is ordered because
    # each panel depends on the factorization of the preceding panel.
    for k in rows:
        active_rows = rows[k:256]
        column = LU[active_rows, k]
        _, pivot_offset = I.arg_reduce.max(I.abs(column), axis=0)
        pivot_row = k + pivot_offset

        if pivot_row != k:
            for j in I.parallel(augmented_cols):
                saved = LU[k, j]
                LU[k, j] = LU[pivot_row, j]
                LU[pivot_row, j] = saved

        pivot = LU[k, k]
        below = rows[k + 1 : 256]
        trailing = cols[k + 1 : 256]
        for i in I.parallel(below):
            multiplier = LU[i, k] / pivot
            LU[i, k] = multiplier
            for j in I.parallel(trailing):
                LU[i, j] = LU[i, j] - multiplier * LU[k, j]

    # Forward substitution for L*y=B, kept in the augmented RHS column.
    for k in rows:
        pivot_rhs = LU[k, 256]
        below = rows[k + 1 : 256]
        for i in I.parallel(below):
            LU[i, 256] = LU[i, 256] - LU[i, k] * pivot_rhs


@intent.kernel
def lu_backsolve(
    LU: I.InOut[I.f32, (256, 257)],
    X: I.Out[I.f32, (256, 1)],
):
    rows = I.domain(0, 256)
    cols = I.domain(0, 256)

    # Back substitution consumes solved RHS entries from bottom to top.
    for ordinal in I.domain(0, 256):
        k = 255 - ordinal
        trailing = cols[k + 1 : 256]
        correction = I.dot(LU[k, trailing], LU[trailing, 256], acc_dtype=I.f32)
        LU[k, 256] = (LU[k, 256] - correction) / LU[k, k]

    for i in I.parallel(rows):
        X[i, 0] = LU[i, 256]


def build(context):
    factor_artifact = context.compile("lu_factor", lu_factor, constexprs={})
    backsolve_artifact = context.compile("lu_backsolve", lu_backsolve, constexprs={})

    def solve(A, B):
        lu = torch.empty((N, N + 1), device=A.device, dtype=A.dtype)
        out = torch.empty_like(B)
        factor_artifact(A, B, lu)
        backsolve_artifact(lu, out)
        return out

    return solve
