from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def initialize_columns(
    source: I.In[I.f32, ("M", "N")],
    destination: I.Out[I.f32, ("M", "N")],
) -> None:
    rows = I.domain(0, source.shape[0])
    columns = I.domain(0, source.shape[1])
    destination[rows, columns] = source[rows, columns]


@intent.kernel
def jacobi_sweep(values: I.InOut[I.f32, ("M", "N")]) -> None:
    row_count, column_count = values.shape
    rows = I.domain(0, row_count)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    two = I.cast(2.0, I.f32)
    epsilon = I.cast(1.0e-14, I.f32)

    for first in range(0, column_count - 1):
        for second in range(first + 1, column_count):
            left = values[rows, first]
            right = values[rows, second]
            left_norm = I.dot(left, left, acc_dtype=I.f32)
            right_norm = I.dot(right, right, acc_dtype=I.f32)
            cross_product = I.dot(left, right, acc_dtype=I.f32)
            column_scale = I.sqrt(left_norm * right_norm)
            rotation_threshold = epsilon * column_scale
            should_rotate = I.abs(cross_product) > rotation_threshold

            zeta = (right_norm - left_norm) / (two * cross_product)
            zeta_magnitude = I.abs(zeta)
            sign = I.select(zeta < zero, -one, one)
            raw_tangent = sign / (zeta_magnitude + I.sqrt(one + zeta * zeta))
            tangent = I.select(should_rotate, raw_tangent, zero)
            cosine = one / I.sqrt(one + tangent * tangent)
            sine = cosine * tangent

            values[rows, first] = cosine * left - sine * right
            values[rows, second] = sine * left + cosine * right


@intent.kernel
def reconstruct_columns(
    orthogonalized: I.In[I.f32, ("M", "N")],
    reconstructed: I.Out[I.f32, ("M", "N")],
) -> None:
    rows = I.domain(0, orthogonalized.shape[0])
    zero = I.cast(0.0, I.f32)

    for column in I.parallel(I.domain(0, orthogonalized.shape[1])):
        values = orthogonalized[:, column]
        squared_norm = I.dot(values, values, acc_dtype=I.f32)
        is_nonzero = squared_norm > zero
        normalization = I.select(is_nonzero, I.rsqrt(squared_norm), zero)
        reconstructed[rows, column] = normalization * values


def build(context):
    initialize = context.compile(
        "fused_svd_initialize_columns", initialize_columns
    )
    sweep = context.compile("fused_svd_jacobi_sweep", jacobi_sweep)
    reconstruct = context.compile(
        "fused_svd_reconstruct_columns", reconstruct_columns
    )

    def fused_svd_reconstruct(A: "torch.Tensor") -> "torch.Tensor":
        orthogonalized = A.new_empty(A.shape)
        reconstructed = A.new_empty(A.shape)
        initialize(A, orthogonalized)
        for _ in range(12):
            sweep(orthogonalized)
        reconstruct(orthogonalized, reconstructed)
        return reconstructed

    return fused_svd_reconstruct
