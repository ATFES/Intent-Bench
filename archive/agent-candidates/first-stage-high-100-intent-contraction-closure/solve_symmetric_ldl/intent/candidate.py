import torch
import intent
import intent.language as I


@intent.kernel
def _factor_kernel(
    A: I.In[I.f32, (32, 32)],
    F: I.InOut[I.f32, (32, 32)],
    pivots: I.Out[I.i64, (32,)],
):
    zero = I.cast(0.0, I.f32)
    alpha = I.cast(0.6403882, I.f32)

    # Materialize the lower-triangle input as the symmetric working matrix.
    for i in range(32):
        for j in range(32):
            if i >= j:
                F[i, j] = I.cast(A[i, j], I.f32)
            else:
                F[i, j] = I.cast(A[j, i], I.f32)

    k = I.indices(I.domain(0, 1))[0]
    while k < 31:
        absakk = I.abs(F[k, k])
        colmax = zero
        imax = k + 1
        for i in range(32):
            if i > k:
                value = I.abs(F[i, k])
                if value > colmax:
                    colmax = value
                    imax = i

        kp = k
        use_two = False
        if colmax != zero:
            if absakk < alpha * colmax:
                rowmax = zero
                for j in range(32):
                    if j >= k and j != imax:
                        value = I.abs(F[imax, j])
                        if value > rowmax:
                            rowmax = value
                if I.abs(F[imax, imax]) < alpha * rowmax:
                    if absakk < alpha * colmax * (colmax / rowmax):
                        kp = imax
                        use_two = True
                else:
                    kp = imax

        if use_two:
            # A 2x2 pivot is placed at k,k+1; LAPACK records -kp (one based).
            if kp != k + 1:
                for j in range(32):
                    if j < k:
                        tmp = F[k + 1, j]
                        F[k + 1, j] = F[kp, j]
                        F[kp, j] = tmp
                for j in range(32):
                    if j >= k:
                        tmp = F[k + 1, j]
                        F[k + 1, j] = F[kp, j]
                        F[kp, j] = tmp
                for i in range(32):
                    if i >= k:
                        tmp = F[i, k + 1]
                        F[i, k + 1] = F[i, kp]
                        F[i, kp] = tmp

            pivots[k] = I.cast(-(kp + 1), I.i64)
            pivots[k + 1] = I.cast(-(kp + 1), I.i64)

            d00 = F[k, k]
            d10 = F[k + 1, k]
            d11 = F[k + 1, k + 1]
            det = d00 * d11 - d10 * d10

            for i in range(32):
                if i > k + 1:
                    x0 = F[i, k]
                    x1 = F[i, k + 1]
                    F[i, k] = (x0 * d11 - x1 * d10) / det
                    F[i, k + 1] = (x1 * d00 - x0 * d10) / det

            for i in range(32):
                if i > k + 1:
                    t0 = F[i, k] * d00 + F[i, k + 1] * d10
                    t1 = F[i, k] * d10 + F[i, k + 1] * d11
                    for j in range(32):
                        if j > k + 1:
                            F[i, j] = F[i, j] - t0 * F[j, k] - t1 * F[j, k + 1]
            k = k + 2
        else:
            if kp != k:
                for j in range(32):
                    if j < k:
                        tmp = F[k, j]
                        F[k, j] = F[kp, j]
                        F[kp, j] = tmp
                for j in range(32):
                    if j >= k:
                        tmp = F[k, j]
                        F[k, j] = F[kp, j]
                        F[kp, j] = tmp
                for i in range(32):
                    if i >= k:
                        tmp = F[i, k]
                        F[i, k] = F[i, kp]
                        F[i, kp] = tmp

            pivots[k] = I.cast(kp + 1, I.i64)
            diag = F[k, k]
            for i in range(32):
                if i > k:
                    F[i, k] = F[i, k] / diag
            for i in range(32):
                if i > k:
                    for j in range(32):
                        if j > k:
                            F[i, j] = F[i, j] - F[i, k] * diag * F[j, k]
            k = k + 1

    if k == 31:
        pivots[31] = I.cast(32, I.i64)

    # The public compact factor contains no strictly upper-triangular values.
    for i in range(32):
        for j in range(32):
            if i < j:
                F[i, j] = zero


@intent.kernel
def _reconstruct_solve_kernel(
    F: I.In[I.f32, (32, 32)],
    pivots: I.In[I.i64, (32,)],
    b: I.In[I.f32, (32, 1)],
    C: I.Out[I.f32, (32, 32)],
    out: I.Out[I.f32, (32, 1)],
):
    zero = I.cast(0.0, I.f32)

    # Reconstruct C = F diag(float32(p)) F^T from the compact factor.
    for i in range(32):
        for j in range(32):
            value = zero
            for k in range(32):
                if k <= i and k <= j:
                    value = value + F[i, k] * I.cast(pivots[k], I.f32) * F[j, k]
            C[i, j] = value

    for i in range(32):
        out[i, 0] = I.cast(b[i, 0], I.f32)

    # In-place Gaussian elimination with complete row pivoting on the reconstructed C.
    for k in range(31):
        pivot = k
        best = I.abs(C[k, k])
        for i in range(32):
            if i > k:
                value = I.abs(C[i, k])
                if value > best:
                    best = value
                    pivot = i

        if pivot != k:
            for j in range(32):
                tmp = C[k, j]
                C[k, j] = C[pivot, j]
                C[pivot, j] = tmp
            tmp_rhs = out[k, 0]
            out[k, 0] = out[pivot, 0]
            out[pivot, 0] = tmp_rhs

        diag = C[k, k]
        for i in range(32):
            if i > k:
                multiplier = C[i, k] / diag
                C[i, k] = zero
                for j in range(32):
                    if j > k:
                        C[i, j] = C[i, j] - multiplier * C[k, j]

    for reverse in range(32):
        i = 31 - reverse
        value = out[i, 0]
        for j in range(32):
            if j > i:
                value = value - C[i, j] * out[j, 0]
        out[i, 0] = value / C[i, i]


def build(context):
    factor = context.compile("symmetric_ldl_factor", _factor_kernel)
    reconstruct_solve = context.compile("symmetric_ldl_reconstruct_solve", _reconstruct_solve_kernel)

    def solve_symmetric_ldl(A, b, hermitian=False, out=None):
        del hermitian
        if out is None:
            out = torch.empty((32, 1), dtype=torch.float32, device=A.device)
        F = torch.empty((32, 32), dtype=torch.float32, device=A.device)
        pivots = torch.empty((32,), dtype=torch.int64, device=A.device)
        C = torch.empty((32, 32), dtype=torch.float32, device=A.device)
        factor(A, F, pivots)
        reconstruct_solve(F, pivots, b, C, out)
        return out

    return solve_symmetric_ldl
