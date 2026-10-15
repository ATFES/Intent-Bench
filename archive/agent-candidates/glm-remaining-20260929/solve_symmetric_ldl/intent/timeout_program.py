import torch
import intent
import intent.language as I


@intent.kernel
def _ldl_factor_kernel(
    F: I.Out[I.f32, ("N", "N")],
    pivots: I.Out[I.i32, ("N",)],
    A: I.In[I.f32, ("N", "N")],
):
    n = A.shape[0]
    work = I.buffer(A.shape, I.f32, init=I.cast(A[:, :], I.f32))
    factors = I.buffer(
        A.shape,
        I.f32,
        init=I.full(A.shape, fill=0.0, dtype=I.f32),
    )
    pivot_values = I.buffer(
        (n,),
        I.i32,
        init=I.full((n,), fill=0, dtype=I.i32),
    )
    alpha = I.cast(0.6403882032022076, I.f32)
    position = I.cast(0, I.index)

    while position < n:
        largest_off = I.cast(0.0, I.f32)
        largest_row = position
        if position + 1 < n:
            off_values = I.abs(work[I.domain(position + 1, n), position])
            largest_off, relative_row = I.arg_reduce.max(off_values, axis=0)
            largest_row = position + 1 + relative_row
            row_columns = I.domain(position + 1, n)
            row_coordinates = I.indices(row_columns)
            row_off_values = I.abs(work[largest_row, row_columns])
            other_off_values = I.mask(
                row_off_values,
                row_coordinates != largest_row,
                I.cast(0.0, I.f32),
            )
            other_off = I.reduce.max(
                other_off_values,
                axis=0,
                acc_dtype=I.f32,
            )
        else:
            other_off = I.cast(0.0, I.f32)

        current_diagonal = I.abs(work[position, position])
        selected_diagonal = I.abs(work[largest_row, largest_row])
        use_first_one_by_one = current_diagonal >= alpha * largest_off
        use_swapped_one_by_one = selected_diagonal >= alpha * other_off

        if use_first_one_by_one or use_swapped_one_by_one:
            if use_swapped_one_by_one:
                other = largest_row
                for row in I.parallel(I.domain(0, n)):
                    temporary = work[row, position]
                    work[row, position] = work[row, other]
                    work[row, other] = temporary
                for column in I.parallel(I.domain(0, n)):
                    temporary = work[position, column]
                    work[position, column] = work[other, column]
                    work[other, column] = temporary
                for previous in I.parallel(I.domain(0, position)):
                    temporary = factors[position, previous]
                    factors[position, previous] = factors[other, previous]
                    factors[other, previous] = temporary
                pivot_values[position] = I.cast(other + 1, I.i32)
            else:
                pivot_values[position] = I.cast(position + 1, I.i32)

            pivot = work[position, position]
            for row in I.parallel(I.domain(position + 1, n)):
                multiplier = work[row, position] / pivot
                factors[row, position] = multiplier
            for row in I.parallel(I.domain(position + 1, n)):
                for column in I.parallel(I.domain(position + 1, n)):
                    update = (
                        factors[row, position]
                        * pivot
                        * factors[column, position]
                    )
                    work[row, column] = work[row, column] - update
            factors[position, position] = pivot
            position = position + 1
        else:
            if largest_row != position + 1:
                adjacent = position + 1
                other = largest_row
                for row in I.parallel(I.domain(0, n)):
                    temporary = work[row, adjacent]
                    work[row, adjacent] = work[row, other]
                    work[row, other] = temporary
                for column in I.parallel(I.domain(0, n)):
                    temporary = work[adjacent, column]
                    work[adjacent, column] = work[other, column]
                    work[other, column] = temporary
                for previous in I.parallel(I.domain(0, position)):
                    temporary = factors[adjacent, previous]
                    factors[adjacent, previous] = factors[other, previous]
                    factors[other, previous] = temporary
            first = work[position, position]
            coupling = work[position + 1, position]
            second = work[position + 1, position + 1]
            determinant = first * second - coupling * coupling
            inverse_first_second = -coupling / determinant
            inverse_first = second / determinant
            inverse_second = first / determinant
            for row in I.parallel(I.domain(position + 2, n)):
                right_first = work[row, position]
                right_second = work[row, position + 1]
                factors[row, position] = (
                    inverse_first * right_first
                    + inverse_first_second * right_second
                )
                factors[row, position + 1] = (
                    inverse_first_second * right_first
                    + inverse_second * right_second
                )
            for row in I.parallel(I.domain(position + 2, n)):
                for column in I.parallel(I.domain(position + 2, n)):
                    row_first = factors[row, position]
                    row_second = factors[row, position + 1]
                    column_first = factors[column, position]
                    column_second = factors[column, position + 1]
                    update = (
                        row_first * first * column_first
                        + row_first * coupling * column_second
                        + row_second * coupling * column_first
                        + row_second * second * column_second
                    )
                    work[row, column] = work[row, column] - update
            factors[position, position] = first
            factors[position + 1, position] = coupling
            factors[position + 1, position + 1] = second
            pivot_values[position] = I.cast(-(position + 1), I.i32)
            pivot_values[position + 1] = I.cast(position + 2, I.i32)
            position = position + 2

    F[:, :] = factors[:, :]
    pivots[:] = pivot_values[:]


@intent.kernel
def _reconstruct_kernel(
    C: I.Out[I.f32, ("N", "N")],
    F: I.In[I.f32, ("N", "N")],
    pivots: I.In[I.i32, ("N",)],
):
    n = F.shape[0]
    f_values = F[:, :]
    diagonal = I.reshape(I.cast(pivots[:], I.f32), (n, 1))
    scaled = f_values * diagonal
    C[:, :] = I.matmul(
        scaled,
        f_values,
        acc_dtype=I.f32,
        transpose_rhs=True,
    )


@intent.kernel
def _solve_kernel(
    X: I.Out[I.f32, ("N", "M")],
    C: I.In[I.f32, ("N", "N")],
    B: I.In[I.f32, ("N", "M")],
):
    n = C.shape[0]
    m = B.shape[1]
    triangular = I.buffer(
        C.shape,
        I.f32,
        init=I.full(C.shape, fill=0.0, dtype=I.f32),
    )
    right_hand_side = I.buffer(
        B.shape,
        I.f32,
        init=I.cast(B[:, :], I.f32),
    )
    solution = I.buffer(
        B.shape,
        I.f32,
        init=I.full(B.shape, fill=0.0, dtype=I.f32),
    )

    for column in I.domain(0, n):
        prior = I.domain(0, column)
        prior_values = triangular[column, prior]
        energy = I.reduce.sum(
            prior_values * prior_values,
            axis=0,
            acc_dtype=I.f32,
        )
        diagonal_value = I.sqrt(C[column, column] - energy)
        triangular[column, column] = diagonal_value
        for row in I.parallel(I.domain(column + 1, n)):
            row_values = triangular[row, prior]
            product = I.reduce.sum(
                row_values * prior_values,
                axis=0,
                acc_dtype=I.f32,
            )
            triangular[row, column] = (C[row, column] - product) / diagonal_value

    for row in I.domain(0, n):
        prior = I.domain(0, row)
        intermediate = right_hand_side[row, :]
        if row > 0:
            products = triangular[row, prior] * right_hand_side[prior, :]
            intermediate = intermediate - I.reduce.sum(
                products,
                acc_dtype=I.f32,
                axis=0,
            )
        intermediate = I.reshape(intermediate, (m,))
        right_hand_side[row, :] = intermediate / triangular[row, row]

    for reverse_ordinal in I.domain(0, n):
        row = n - 1 - reverse_ordinal
        suffix = I.domain(row + 1, n)
        result = right_hand_side[row, :]
        if row + 1 < n:
            products = triangular[suffix, row] * solution[suffix, :]
            result = result - I.reduce.sum(
                products,
                acc_dtype=I.f32,
                axis=0,
            )
        result = result / triangular[row, row]
        result = I.reshape(result, (m,))
        solution[row, :] = result
        right_hand_side[row, :] = result

    X[:, :] = solution[:, :]


@intent.kernel
def _copy_kernel(
    destination: I.Out[I.f32, ("N", "M")],
    source: I.In[I.f32, ("N", "M")],
):
    destination[:, :] = source[:, :]


def build(context):
    factor = context.compile("solve_ldl_factor", _ldl_factor_kernel)
    reconstruct = context.compile("solve_ldl_reconstruct", _reconstruct_kernel)
    solve = context.compile("solve_ldl_solve", _solve_kernel)
    copy_result = context.compile("solve_ldl_copy_result", _copy_kernel)

    def solve_symmetric_ldl(A, b, hermitian=False, out=None):
        device = A.device
        factor_tensor = torch.empty(
            (32, 32), device=device, dtype=torch.float32
        )
        pivot_tensor = torch.empty((32,), device=device, dtype=torch.int32)
        reconstructed = torch.empty(
            (32, 32), device=device, dtype=torch.float32
        )
        solution = torch.empty((32, 1), device=device, dtype=torch.float32)
        factor(factor_tensor, pivot_tensor, A)
        reconstruct(reconstructed, factor_tensor, pivot_tensor)
        solve(solution, reconstructed, b)
        if out is None:
            return solution
        copy_result(out, solution)
        return out

    return solve_symmetric_ldl
