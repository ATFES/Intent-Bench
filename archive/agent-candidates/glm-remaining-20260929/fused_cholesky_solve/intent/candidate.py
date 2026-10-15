import intent
import intent.language as I
import torch


@intent.kernel
def cholesky_factor(
    a: I.In[I.f32, ("N", "N")],
    factor: I.Out[I.f32, ("N", "N")],
):
    size = a.shape[0]
    factor_buffer = I.buffer((size, size), I.f32)

    for column in I.domain(0, size):
        previous = I.domain(0, column)
        diagonal_row = factor_buffer[column, previous]
        diagonal_value = a[column, column] - I.dot(
            diagonal_row,
            diagonal_row,
            acc_dtype=I.f32,
        )
        diagonal = I.sqrt(diagonal_value)
        factor_buffer[column, column] = diagonal
        factor[column, column] = diagonal

        for row in I.parallel(I.domain(column + 1, size)):
            row_values = factor_buffer[row, previous]
            column_values = factor_buffer[column, previous]
            residual = a[row, column] - I.dot(
                row_values,
                column_values,
                acc_dtype=I.f32,
            )
            column_value = I.fdiv(residual, diagonal)
            factor_buffer[row, column] = column_value
            factor[row, column] = column_value


@intent.kernel
def forward_substitute(
    b: I.In[I.f32, ("N", "K")],
    factor: I.In[I.f32, ("N", "N")],
    solution: I.Out[I.f32, ("N", "K")],
):
    row_count = b.shape[0]
    column_count = b.shape[1]
    solution_buffer = I.buffer((row_count, column_count), I.f32)

    for row in I.domain(0, row_count):
        previous = I.domain(0, row)
        factor_values = factor[row, previous]
        for column in I.parallel(I.domain(0, column_count)):
            previous_solution = solution_buffer[previous, column]
            partial_sum = I.dot(
                factor_values,
                previous_solution,
                acc_dtype=I.f32,
            )
            solution_buffer[row, column] = b[row, column] - partial_sum

    solution[:, :] = solution_buffer[:, :]


@intent.kernel
def backward_substitute(
    factor: I.In[I.f32, ("N", "N")],
    solution: I.InOut[I.f32, ("N", "K")],
):
    row_count = solution.shape[0]
    column_count = solution.shape[1]

    for ordinal in I.domain(0, row_count):
        row = row_count - 1 - ordinal
        previous = I.domain(row + 1, row_count)
        factor_values = factor[row, previous]
        for column in I.parallel(I.domain(0, column_count)):
            later_solution = solution[previous, column]
            partial_sum = I.dot(
                factor_values,
                later_solution,
                acc_dtype=I.f32,
            )
            diagonal = factor[row, row]
            solution[row, column] = I.fdiv(
                solution[row, column] - partial_sum,
                diagonal,
            )


def build(context):
    factor_kernel = context.compile("fused_cholesky_factor", cholesky_factor)
    forward_kernel = context.compile("fused_forward_substitute", forward_substitute)
    backward_kernel = context.compile(
        "fused_backward_substitute",
        backward_substitute,
    )

    def fused_cholesky_solve(a, b):
        factor = torch.empty(a.shape, dtype=a.dtype, device=a.device)
        solution = torch.empty(b.shape, dtype=b.dtype, device=b.device)
        factor_kernel(a, factor)
        forward_kernel(b, factor, solution)
        backward_kernel(factor, solution)
        return solution

    return fused_cholesky_solve
