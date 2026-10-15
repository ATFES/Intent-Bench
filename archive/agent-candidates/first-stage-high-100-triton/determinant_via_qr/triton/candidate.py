import torch
import triton
import triton.language as tl


@triton.jit
def _determinant_qr_kernel(
    a_ptr,
    out_ptr,
    stride_a0,
    stride_a1,
    N: tl.constexpr,
):
    # The complete 32x32 problem is kept in one program.  Applying Householder
    # reflectors gives an R factor without forming Q explicitly.  Since N is
    # even for this task, the product of R's diagonal is det(A).
    rows = tl.arange(0, N)
    cols = tl.arange(0, N)
    work = tl.load(
        a_ptr + rows[:, None] * stride_a0 + cols[None, :] * stride_a1
    )
    determinant = 1.0

    for k in range(N):
        row_active = rows >= k
        col_active = cols >= k

        # Select column k from the register-resident matrix.  The explicit
        # reduction avoids dynamic tensor indexing in the generated kernel.
        x = tl.sum(
            tl.where((cols == k)[None, :], work, 0.0),
            axis=1,
        )
        x = tl.where(row_active, x, 0.0)

        norm_sq = tl.sum(x * x, axis=0)
        norm = tl.sqrt(norm_sq)
        x0 = tl.sum(tl.where(rows == k, x, 0.0), axis=0)
        sign = tl.where(x0 >= 0.0, 1.0, -1.0)
        alpha = -sign * norm

        v = tl.where(rows == k, x - alpha, x)
        v_norm_sq = tl.sum(v * v, axis=0)
        nonzero = v_norm_sq > 0.0
        safe_v_norm_sq = tl.where(nonzero, v_norm_sq, 1.0)
        beta = tl.where(nonzero, 2.0 / safe_v_norm_sq, 0.0)

        projection = tl.sum(v[:, None] * work, axis=0)
        active = row_active[:, None] & col_active[None, :]
        work = tl.where(
            active,
            work - beta * v[:, None] * projection[None, :],
            work,
        )
        determinant = determinant * alpha

    tl.store(out_ptr, determinant)


def build(context):
    def wrapper(A, *, mode="reduced", out=None):
        if out is None:
            out = torch.empty((), device=A.device, dtype=A.dtype)
        _determinant_qr_kernel[(1,)](
            A,
            out,
            A.stride(0),
            A.stride(1),
            N=32,
            num_warps=8,
        )
        return out

    return wrapper
