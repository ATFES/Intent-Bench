import torch
import intent
import intent.language as I


@intent.kernel
def qr_det_kernel(A: I.In[I.f32, ("N", "N")], out: I.Out[I.f32, ()]):
    n = A.shape[0]
    all_rows = I.domain(0, n)
    all_cols = I.domain(0, n)

    # Householder QR is performed in place in a kernel-local logical buffer.
    r = I.buffer((n, n), I.f32, init=A[all_rows, all_cols])

    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    two = I.cast(2.0, I.f32)

    # det(Q) is accumulated from the number of non-trivial Householder
    # reflectors: each one contributes -1 for real matrices.
    q_sign = one

    for k in I.domain(0, n):
        rows = I.domain(k, n)
        col = r[rows, k]
        x0 = r[k, k]

        norm = I.sqrt(I.reduce.sum(col * col, axis=0))
        alpha = I.select(x0 >= zero, -norm, norm)

        ids = I.indices(rows)
        v = I.select(ids == k, x0 - alpha, col)

        vdot = I.reduce.sum(v * v, axis=0)
        nonzero = vdot > zero
        safe_vdot = I.select(nonzero, vdot, one)
        tau = I.fdiv(two, safe_vdot)
        tau = I.select(nonzero, tau, zero)

        q_sign = I.select(norm > zero, -q_sign, q_sign)

        # Write the reflection result into column k: R[k,k] = alpha and
        # all entries below the diagonal are zero.
        r[k, k] = alpha
        below = I.domain(k + 1, n)
        r[below, k] = zero

        # Apply the same Householder reflection to the trailing columns.
        trail_cols = I.domain(k + 1, n)
        block = r[rows, trail_cols]
        w = I.vecmat(v, block, acc_dtype=I.f32)
        r[rows, trail_cols] = block - tau * I.outer(v, w)

    # Product of R diagonal entries, then account for det(Q).
    det = one
    for i in I.domain(0, n):
        det = det * r[i, i]

    out[()] = det * q_sign


def build(context):
    qr_det = context.compile("determinant_via_qr_householder", qr_det_kernel)

    def determinant_via_qr(A, *, mode="reduced", out=None):
        result = out
        if result is None:
            result = torch.empty((), dtype=A.dtype, device=A.device)
        qr_det(A, result)
        return result

    return determinant_via_qr
