import torch
import triton
import triton.language as tl


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _iter_batch(shape):
    if len(shape) == 0:
        yield ()
        return
    first = shape[0]
    rest = shape[1:]
    for i in range(first):
        for tail in _iter_batch(rest):
            yield (i,) + tail


@triton.jit
def _qr_lstsq_kernel(
    A_ptr,
    B_ptr,
    X_ptr,
    stride_am,
    stride_an,
    stride_bm,
    stride_bk,
    stride_xm,
    stride_xk,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_mask = (rm[:, None] < M) & (rn[None, :] < N)
    a = tl.load(A_ptr + rm[:, None] * stride_am + rn[None, :] * stride_an,
                mask=a_mask, other=0.0)

    b_mask = (rm[:, None] < M) & (rk[None, :] < K)
    b = tl.load(B_ptr + rm[:, None] * stride_bm + rk[None, :] * stride_bk,
                mask=b_mask, other=0.0)

    # Householder QR, applied simultaneously to b so that b becomes Q^H b.
    for j in tl.static_range(N):
        colmask = (rm[:, None] < M) & (rn[None, :] == j)
        x = tl.sum(tl.where(colmask, a, 0.0), axis=1)

        rowmask = (rm >= j) & (rm < M)
        x2 = x[:, None] * x[:, None]
        norm2 = tl.sum(tl.where(rowmask[:, None], x2, 0.0))
        norm = tl.sqrt_rn(norm2)

        xj = tl.sum(tl.where(rm == j, x, 0.0))
        sign = tl.where(xj >= 0.0, 1.0, -1.0)
        alpha = xj + sign * norm

        v = tl.where(rowmask[:, None], x[:, None], 0.0)
        v = tl.where(rm[:, None] == j, alpha, v)

        vnorm2 = tl.sum(v * v)
        beta = 2.0 / tl.maximum(vnorm2, 1.0e-20)

        dots = tl.sum(v * a, axis=0)
        upd = beta * v * dots[None, :]
        upd_mask = (rm[:, None] < M) & (rn[None, :] >= j) & (rn[None, :] < N)
        a = tl.where(upd_mask, a - upd, a)

        db = tl.sum(v * b, axis=0)
        b = b - beta * v * db[None, :]

    # Back substitution: solve upper-triangular R x = (Q^H b)[0:N].
    x = tl.zeros([BLOCK_N, BLOCK_K], tl.float32)
    for it in tl.static_range(N):
        row = N - 1 - it
        rrow_mask = (rm[:, None] == row) & (rn[None, :] < N)
        rrow = tl.sum(tl.where(rrow_mask, a, 0.0), axis=0)

        dot_mask = (rn[:, None] > row) & (rk[None, :] < K)
        dot = tl.sum(tl.where(dot_mask, rrow[:, None] * x, 0.0), axis=0)
        qrow_mask = (rm[:, None] == row) & (rk[None, :] < K)
        qrow = tl.sum(tl.where(qrow_mask, b, 0.0), axis=0)
        diag = tl.sum(tl.where(rn == row, rrow, 0.0))
        xrow = (qrow - dot) / diag
        xmask = (rn[:, None] == row) & (rk[None, :] < K)
        x = tl.where(xmask, xrow[None, :], x)

    x_mask = (rn[:, None] < N) & (rk[None, :] < K)
    tl.store(X_ptr + rn[:, None] * stride_xm + rk[None, :] * stride_xk,
             x, mask=x_mask)


def least_squares_qr(A, b, *, mode='reduced', out=None):
    if mode not in ('reduced', 'complete'):
        raise ValueError("mode must be either 'reduced' or 'complete'")
    if A.dim() < 2:
        raise ValueError("A must have shape (*, m, n)")
    if A.device != b.device:
        raise ValueError("A and b must be on the same device")
    if A.dtype != b.dtype:
        raise ValueError("A and b must have the same dtype")
    if A.dtype != torch.float32:
        raise NotImplementedError("only float32 is supported")

    m = A.shape[-2]
    n = A.shape[-1]
    batch_shape = A.shape[:-2]

    if m < n:
        raise ValueError("least_squares_qr requires an overdetermined system (m >= n)")

    if b.dim() == A.dim() - 1:
        vector_rhs = True
        k = 1
        expected_b = batch_shape + (m,)
    elif b.dim() == A.dim():
        vector_rhs = False
        k = b.shape[-1]
        expected_b = batch_shape + (m, k)
    else:
        raise ValueError("b must have shape (*, m) or (*, m, k)")

    if b.shape != expected_b:
        raise ValueError(f"expected b shape {expected_b}, got {tuple(b.shape)}")

    out_shape = batch_shape + ((n,) if vector_rhs else (n, k))
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f"expected out shape {out_shape}, got {tuple(out.shape)}")
        if out.dtype != A.dtype:
            raise ValueError("out dtype must match A dtype")
        if out.device != A.device:
            raise ValueError("out must be on the same device as A")
        result = out
    else:
        result = torch.empty(out_shape, device=A.device, dtype=A.dtype)

    block_m = _next_pow2(m)
    block_n = _next_pow2(n)
    block_k = _next_pow2(k)

    has_batch = len(batch_shape) > 0
    if has_batch:
        for idx in _iter_batch(batch_shape):
            a_i = A[idx]
            b_i = b[idx]
            if vector_rhs:
                bmat = b_i.unsqueeze(-1)
                x_out = result[idx].unsqueeze(-1)
            else:
                bmat = b_i
                x_out = result[idx]

            _qr_lstsq_kernel[(1,)](
                a_i, bmat, x_out,
                a_i.stride(0), a_i.stride(1),
                bmat.stride(0), bmat.stride(1),
                x_out.stride(0), x_out.stride(1),
                M=m, N=n, K=k,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            )
    else:
        if vector_rhs:
            bmat = b.unsqueeze(-1)
            x_out = result.unsqueeze(-1)
        else:
            bmat = b
            x_out = result

        _qr_lstsq_kernel[(1,)](
            A, bmat, x_out,
            A.stride(0), A.stride(1),
            bmat.stride(0), bmat.stride(1),
            x_out.stride(0), x_out.stride(1),
            M=m, N=n, K=k,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )

    return result


def build(context):
    return least_squares_qr
