import torch
import intent
import intent.language as I


@intent.kernel
def symmetric_partials(
    A: I.In[I.f32, ("G", 32, "K")],
    C: I.In[I.f32, ("G", 32, "G", 32)],
    partials: I.Out[I.f32, ("G", "G"), I.constraints(noalias=True)],
    alpha: I.f32,
    beta: I.f32,
):
    groups = I.domain(0, A.shape[0])
    members = I.domain(0, 32)
    reduction = I.domain(0, A.shape[2])
    for row in I.parallel(groups):
        for column in I.parallel(groups):
            if row <= column:
                product = I.matmul(
                    A[row, members, reduction], A[column, members, reduction],
                    transpose_rhs=True, acc_dtype=I.f32,
                )
                upper = alpha * product + beta * C[row, members, column, members]
                value = I.reduce.sum(I.abs(upper), axis=(0, 1), acc_dtype=I.f32)
                if row < column:
                    lower = alpha * I.transpose(product, (1, 0)) + beta * C[column, members, row, members]
                    value = value + I.reduce.sum(I.abs(lower), axis=(0, 1), acc_dtype=I.f32)
                partials[row, column] = value
            else:
                partials[row, column] = 0.0


@intent.kernel
def finish_sum(
    partials: I.In[I.f32, ("G", "G")],
    result: I.Out[I.f32, (), I.constraints(noalias=True)],
):
    result[()] = I.reduce.sum(partials[:, :], axis=(0, 1), acc_dtype=I.f32)


def build(context):
    partial = context.compile("symmetric_partials", symmetric_partials)
    finish = context.compile("finish_sum", finish_sum)

    def symmetric_mm_and_abs_sum(A, C, alpha=1.0, beta=0.5):
        rows, width = A.shape
        if rows % 32 or C.shape != (rows, rows):
            raise NotImplementedError("this grouped implementation requires complete groups of 32 rows and square C")
        groups = rows // 32
        partials = torch.empty((groups, groups), dtype=A.dtype, device=A.device)
        result = torch.empty((), dtype=A.dtype, device=A.device)
        partial(A.view(groups, 32, width), C.view(groups, 32, groups, 32),
                partials, alpha, beta)
        finish(partials, result)
        return result

    return symmetric_mm_and_abs_sum
