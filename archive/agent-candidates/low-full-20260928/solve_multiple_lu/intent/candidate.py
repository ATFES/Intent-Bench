import torch
import intent
import intent.language as I


@intent.kernel
def _solve(A: I.In[I.f32, (64, 64)],
           Bs: I.In[I.f32, (64, 64)],
           Out: I.Out[I.f32, (64, 64)],
           PIVOT: I.Constexpr[bool]):
    lu = I.buffer((64, 64), I.f32, init=0.0)
    rhs = I.buffer((64, 64), I.f32, init=0.0)

    for i in range(64):
        for j in range(64):
            lu[i, j] = A[i, j]
            rhs[i, j] = Bs[i, j]

    for k in range(64):
        pivot_row = k
        if PIVOT:
            column = lu[:, k]
            _, pivot_row = I.arg_reduce.max(I.abs(column), axis=0)

        if PIVOT:
            for j in range(64):
                a = lu[k, j]
                b = lu[pivot_row, j]
                lu[k, j] = b
                lu[pivot_row, j] = a
            for j in range(64):
                a = rhs[k, j]
                b = rhs[pivot_row, j]
                rhs[k, j] = b
                rhs[pivot_row, j] = a

        diagonal = lu[k, k]
        for i in range(k + 1, 64):
            factor = lu[i, k] / diagonal
            lu[i, k] = factor
            for j in range(k + 1, 64):
                lu[i, j] = lu[i, j] - factor * lu[k, j]
            for j in range(64):
                rhs[i, j] = rhs[i, j] - factor * rhs[k, j]

    for kk in range(64):
        k = 63 - kk
        diagonal = lu[k, k]
        for j in range(64):
            value = rhs[k, j]
            for q in range(k + 1, 64):
                value = value - lu[k, q] * rhs[q, j]
            rhs[k, j] = value / diagonal

    for i in range(64):
        for j in range(64):
            Out[i, j] = rhs[i, j]


def build(context):
    with_pivot = context.compile("solve_multiple_lu_pivot", _solve,
                                 constexprs={"PIVOT": True})
    without_pivot = context.compile("solve_multiple_lu_nopivot", _solve,
                                    constexprs={"PIVOT": False})

    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        if out is None:
            out = torch.empty_like(Bs)
        if pivot:
            with_pivot(A, Bs, out)
        else:
            without_pivot(A, Bs, out)
        return out

    return solve_multiple_lu
