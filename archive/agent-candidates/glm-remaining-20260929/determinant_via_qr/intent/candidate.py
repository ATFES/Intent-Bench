from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _determinant_householder_qr(
    A: I.In[I.f32, ("N", "N")],
    determinant: I.Out[I.f32, ()],
) -> None:
    n = A.shape[0]
    working = I.buffer(A.shape, I.f64, init=I.cast(A, I.f64))

    log_absolute_determinant = I.cast(0.0, I.f64)
    determinant_sign = I.cast(1.0, I.f32)
    singular = I.cast(0.0, I.f32)

    for k in I.domain(0, n - 1):
        tail_rows = I.domain(k, n)
        tail_columns = I.domain(k, n)

        column_values = I.cast(working[tail_rows, k], I.f64)
        column_square_sum = I.reduce.sum(
            column_values * column_values,
            axis=0,
            acc_dtype=I.f64,
        )
        column_norm = I.sqrt(column_square_sum)
        diagonal = I.cast(working[k, k], I.f64)
        diagonal_sign = I.cast(
            I.select(diagonal >= 0.0, -1.0, 1.0),
            I.f64,
        )
        reflector_target = diagonal_sign * column_norm
        reflector_norm_squared = column_square_sum - reflector_target * diagonal

        if reflector_norm_squared <= 0.0:
            singular = I.cast(1.0, I.f32)
            break

        reflector_values = column_values - I.full(
            column_values.shape,
            reflector_target,
            I.f64,
        )
        reflection_scale = I.cast(-2.0, I.f64) / reflector_norm_squared

        for column in I.parallel(tail_columns):
            updated_column_values = I.cast(working[tail_rows, column], I.f64)
            projection = I.reduce.sum(
                reflector_values * updated_column_values,
                axis=0,
                acc_dtype=I.f64,
            )
            reflected_values = updated_column_values + (
                reflection_scale
                * projection
                * reflector_values
            )
            working[tail_rows, column] = reflected_values

        reflected_diagonal = I.cast(working[k, k], I.f64)
        reflected_diagonal_abs = I.abs(reflected_diagonal)
        if reflected_diagonal_abs == 0.0:
            singular = I.cast(1.0, I.f32)
            break

        log_absolute_determinant = log_absolute_determinant + I.log(
            reflected_diagonal_abs
        )
        determinant_sign = determinant_sign * I.cast(
            I.select(reflected_diagonal < 0.0, -1.0, 1.0),
            I.f32,
        ) * I.cast(-1.0, I.f32)

    if singular == 0.0:
        final_diagonal = I.cast(working[n - 1, n - 1], I.f64)
        final_diagonal_abs = I.abs(final_diagonal)
        if final_diagonal_abs == 0.0:
            singular = I.cast(1.0, I.f32)
        else:
            log_absolute_determinant = (
                log_absolute_determinant + I.log(final_diagonal_abs)
            )
            determinant_sign = determinant_sign * I.cast(
                I.select(final_diagonal < 0.0, -1.0, 1.0),
                I.f32,
            )

    magnitude = I.cast(I.exp(log_absolute_determinant), I.f32)
    result = I.cast(
        I.select(singular == 0.0, determinant_sign * magnitude, I.cast(0.0, I.f32)),
        I.f32,
    )
    determinant[()] = result


def build(context):
    qr_artifact = context.compile(
        "determinant_householder_qr_fixed_profile",
        _determinant_householder_qr,
    )

    def determinant_via_qr(A, *, mode="reduced", out=None):
        if out is None:
            return qr_artifact.run(A)
        qr_artifact(A, out)
        return out

    return determinant_via_qr
