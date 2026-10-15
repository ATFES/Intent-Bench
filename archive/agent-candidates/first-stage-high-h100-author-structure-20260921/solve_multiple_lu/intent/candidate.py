import torch
import intent
import intent.language as I


@intent.kernel
def _lu_factor(
    A: I.In[I.f32, (64, 64)],
    LU: I.Out[I.f32, (64, 64)],
    P: I.Out[I.i64, (64,)],
    PIVOT: I.Constexpr[bool],
):
    rows = I.domain(0, 64)
    cols = I.domain(0, 64)
    lu = I.buffer((64, 64), I.f32, init=A[rows, cols])

    for i in I.domain(0, 64):
        P[i] = I.cast(i, I.i64)

    for k in I.domain(0, 64):
        if PIVOT:
            column = I.abs(lu[k:64, k])
            _, relative = I.arg_reduce.max(column, 0)
            pivot_row = k + relative
            for j in I.domain(0, 64):
                top = lu[k, j]
                pivot_value = lu[pivot_row, j]
                lu[k, j] = pivot_value
                lu[pivot_row, j] = top
            old_p = P[k]
            pivot_p = P[pivot_row]
            P[k] = pivot_p
            P[pivot_row] = old_p

        diagonal = lu[k, k]
        for i in I.domain(k + 1, 64):
            multiplier = lu[i, k] / diagonal
            lu[i, k] = multiplier
            for j in I.domain(k + 1, 64):
                lu[i, j] = lu[i, j] - multiplier * lu[k, j]

    LU[rows, cols] = lu[:, :]


@intent.kernel
def _triangular_solve(
    LU: I.In[I.f32, (64, 64)],
    P: I.In[I.i64, (64,)],
    Bs: I.In[I.f32, (64, 64)],
    X: I.Out[I.f32, (64, 64)],
    PIVOT: I.Constexpr[bool],
):
    for col in I.parallel(I.domain(0, 64)):
        y = I.buffer((64,), I.f32)

        for i in I.domain(0, 64):
            if PIVOT:
                source_row = I.cast(P[i], I.index)
            else:
                source_row = i
            value = Bs[source_row, col]
            for j in I.domain(0, i):
                value = value - LU[i, j] * y[j]
            y[i] = value

        for ordinal in I.domain(0, 64):
            i = 63 - ordinal
            value = y[i]
            for j in I.domain(i + 1, 64):
                value = value - LU[i, j] * X[j, col]
            X[i, col] = value / LU[i, i]


def build(context):
    factor_no_pivot = context.compile(
        "solve_multiple_lu_factor_no_pivot", _lu_factor, constexprs={"PIVOT": False}
    )
    factor_pivot = context.compile(
        "solve_multiple_lu_factor_pivot", _lu_factor, constexprs={"PIVOT": True}
    )
    solve_no_pivot = context.compile(
        "solve_multiple_lu_solve_no_pivot", _triangular_solve, constexprs={"PIVOT": False}
    )
    solve_pivot = context.compile(
        "solve_multiple_lu_solve_pivot", _triangular_solve, constexprs={"PIVOT": True}
    )

    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        lu = torch.empty_like(A)
        permutation = torch.empty((64,), device=A.device, dtype=torch.int64)
        result = Bs if out is None else out
        if out is None:
            result = torch.empty_like(Bs)

        if pivot:
            factor_pivot(A, lu, permutation)
            solve_pivot(lu, permutation, Bs, result)
        else:
            factor_no_pivot(A, lu, permutation)
            solve_no_pivot(lu, permutation, Bs, result)
        return result

    return solve_multiple_lu
