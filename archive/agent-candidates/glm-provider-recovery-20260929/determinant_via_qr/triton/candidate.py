from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _determinant_via_householder_qr(
    a_ptr,
    out_ptr,
    stride_ar,
    stride_ac,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    matrix_mask = (rows[:, None] < N) & (cols[None, :] < N)
    matrix = tl.load(
        a_ptr + rows[:, None] * stride_ar + cols[None, :] * stride_ac,
        mask=matrix_mask,
        other=0.0,
    )

    determinant = 1.0
    det_q = 1.0
    for k in tl.static_range(N):
        column = tl.sum(tl.where(cols[None, :] == k, matrix, 0.0), axis=1)
        pivot_mask = rows >= k
        pivot = tl.where(pivot_mask, column, 0.0)
        scale = tl.max(tl.abs(pivot), axis=0)
        has_reflector = scale != 0.0

        normalized_scale = tl.where(has_reflector, scale, 1.0)
        normalized = pivot / normalized_scale
        normalized_norm = tl.sqrt(tl.sum(normalized * normalized, axis=0))

        first = tl.sum(tl.where(rows == k, normalized, 0.0), axis=0)
        first_sign = tl.where(first < 0.0, -1.0, 1.0)
        signed_norm = -first_sign * normalized_norm
        reflector = normalized
        reflector = tl.where(rows == k, reflector - signed_norm, reflector)
        reflector_norm2 = tl.sum(reflector * reflector, axis=0)
        safe_reflector_norm2 = tl.where(has_reflector, reflector_norm2, 1.0)

        diagonal = tl.where(has_reflector, scale * signed_norm, 0.0)
        determinant = determinant * diagonal
        det_q = tl.where(has_reflector, -det_q, det_q)

        completed_mask = cols == k
        matrix = tl.where(
            completed_mask[None, :] & (rows > k),
            0.0,
            matrix,
        )
        matrix = tl.where(
            completed_mask[None, :] & (rows == k),
            diagonal,
            matrix,
        )

        projection = tl.sum(reflector[:, None] * matrix, axis=0)
        update = (2.0 / safe_reflector_norm2) * reflector[:, None] * projection[None, :]
        matrix = tl.where(
            has_reflector & (cols[None, :] > k) & matrix_mask,
            matrix - update,
            matrix,
        )

    determinant = det_q * determinant
    tl.store(out_ptr, determinant)


def build(context):
    def determinant_via_qr(A, *, mode="reduced", out=None):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square 2-D tensor")

        if out is None:
            out = torch.empty((), dtype=A.dtype, device=A.device)

        size = A.shape[0]
        block_size = max(16, triton.next_power_of_2(size))
        _determinant_via_householder_qr[(1,)](
            A,
            out,
            A.stride(0),
            A.stride(1),
            N=size,
            BLOCK_N=block_size,
            num_warps=1,
        )
        return out

    return determinant_via_qr
