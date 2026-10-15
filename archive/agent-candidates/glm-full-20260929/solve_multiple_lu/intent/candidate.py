from __future__ import annotations

import intent
import intent.language as I


BLOCK_SIZE = 8


@intent.kernel
def _lu_factor_kernel(
    matrix: I.In[I.f32, ("BATCH", "SIZE", "SIZE")],
    packed_lu: I.Out[I.f32, ("BATCH", "SIZE", "SIZE")],
    row_order: I.Out[I.i64, ("BATCH", "SIZE")],
    pivot: I.bool,
    BLOCK: I.Constexpr[int],
) -> None:
    batch_count, matrix_size, _ = matrix.shape

    for batch in I.parallel(I.domain(0, batch_count)):
        for row in I.parallel(I.domain(0, matrix_size)):
            matrix_columns = I.domain(0, matrix_size)
            packed_lu[batch, row, matrix_columns] = matrix[batch, row, matrix_columns]
            row_order[batch, row] = I.cast(row, I.i64)

        panel_start = I.cast(0, I.index)
        while panel_start < matrix_size:
            panel_width = I.minimum(BLOCK, matrix_size - panel_start)
            panel_end = panel_start + panel_width

            for panel_offset in I.domain(panel_start, panel_end):
                active_rows = I.domain(panel_offset, matrix_size)
                active_column = packed_lu[batch, active_rows, panel_offset]
                _, local_pivot = I.arg_reduce.max(I.abs(active_column), axis=0)
                pivot_row = panel_offset + local_pivot

                if pivot:
                    if pivot_row != panel_offset:
                        for column in I.parallel(I.domain(0, matrix_size)):
                            pivot_value = packed_lu[batch, pivot_row, column]
                            current_value = packed_lu[batch, panel_offset, column]
                            packed_lu[batch, panel_offset, column] = pivot_value
                            packed_lu[batch, pivot_row, column] = current_value

                        pivot_order = row_order[batch, pivot_row]
                        current_order = row_order[batch, panel_offset]
                        row_order[batch, panel_offset] = pivot_order
                        row_order[batch, pivot_row] = current_order

                diagonal = packed_lu[batch, panel_offset, panel_offset]
                for row in I.parallel(I.domain(panel_offset + 1, matrix_size)):
                    multiplier = packed_lu[batch, row, panel_offset] / diagonal
                    packed_lu[batch, row, panel_offset] = multiplier
                    for column in I.parallel(
                        I.domain(panel_offset + 1, panel_end)
                    ):
                        packed_lu[batch, row, column] = (
                            packed_lu[batch, row, column]
                            - multiplier * packed_lu[batch, panel_offset, column]
                        )

            if panel_end < matrix_size:
                trailing_rows = I.domain(panel_end, matrix_size)
                panel_columns = I.domain(panel_start, panel_end)
                trailing_columns = I.domain(panel_end, matrix_size)
                left_panel = packed_lu[batch, trailing_rows, panel_columns]
                upper_panel = packed_lu[batch, panel_columns, trailing_columns]
                panel_update = I.matmul(left_panel, upper_panel, acc_dtype=I.f32)
                update_rows = I.domain(0, matrix_size - panel_end)
                update_columns = I.domain(0, matrix_size - panel_end)
                for row_offset in I.parallel(update_rows):
                    for column_offset in I.parallel(update_columns):
                        row = panel_end + row_offset
                        column = panel_end + column_offset
                        packed_lu[batch, row, column] = (
                            packed_lu[batch, row, column]
                            - panel_update[row_offset, column_offset]
                        )

            panel_start = panel_end


@intent.kernel
def _triangular_solve_kernel(
    packed_lu: I.In[I.f32, ("BATCH", "SIZE", "SIZE")],
    row_order: I.In[I.i64, ("BATCH", "SIZE")],
    right_hand_sides: I.In[I.f32, ("BATCH", "SIZE", "COLUMNS")],
    solution: I.Out[I.f32, ("BATCH", "SIZE", "COLUMNS")],
) -> None:
    batch_count, matrix_size, column_count = right_hand_sides.shape

    for batch in I.parallel(I.domain(0, batch_count)):
        for row in I.parallel(I.domain(0, matrix_size)):
            for column in I.parallel(I.domain(0, column_count)):
                source_row = row_order[batch, row]
                solution[batch, row, column] = right_hand_sides[
                    batch, source_row, column
                ]

        for forward_row in I.domain(0, matrix_size):
            for column in I.parallel(I.domain(0, column_count)):
                prefix_rows = I.domain(0, forward_row)
                lower_terms = (
                    packed_lu[batch, forward_row, prefix_rows]
                    * solution[batch, prefix_rows, column]
                )
                lower_sum = I.reduce.sum(lower_terms, axis=0)
                solution[batch, forward_row, column] = (
                    solution[batch, forward_row, column] - lower_sum
                )

        for reverse_offset in I.domain(0, matrix_size):
            row = matrix_size - 1 - reverse_offset
            for column in I.parallel(I.domain(0, column_count)):
                suffix_rows = I.domain(row + 1, matrix_size)
                upper_terms = (
                    packed_lu[batch, row, suffix_rows]
                    * solution[batch, suffix_rows, column]
                )
                upper_sum = I.reduce.sum(upper_terms, axis=0)
                solution[batch, row, column] = (
                    solution[batch, row, column] - upper_sum
                ) / packed_lu[batch, row, row]


def build(context):
    factor_kernel = context.compile(
        "solve_multiple_lu_factor",
        _lu_factor_kernel,
        constexprs={"BLOCK": BLOCK_SIZE},
    )
    solve_kernel = context.compile(
        "solve_multiple_lu_triangular_solve",
        _triangular_solve_kernel,
    )

    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        matrix_size = A.shape[-1]
        flat_A = A.view(-1, matrix_size, matrix_size)
        flat_Bs = Bs.view(-1, matrix_size, Bs.shape[-1])

        if out is None:
            return solve_kernel.run(
                *factor_kernel.run(flat_A, pivot),
                flat_Bs,
            )

        packed_lu, row_order = factor_kernel.run(flat_A, pivot)
        solve_kernel(packed_lu, row_order, flat_Bs, out)
        return out

    return solve_multiple_lu
