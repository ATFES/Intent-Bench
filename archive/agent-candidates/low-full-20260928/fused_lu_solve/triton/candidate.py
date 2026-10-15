import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lu_solve(src_ptr, lu_ptr, b_ptr, out_ptr, n: tl.constexpr):
    # A single logical program owns the factorization so that each pivot/update
    # observes the preceding step.  The vectors below are the independent row
    # and column work for each trailing update.
    pid = tl.program_id(0)
    if pid == 0:
        rows = tl.arange(0, n)
        cols = tl.arange(0, n)
        full_mask = (rows[:, None] < n) & (cols[None, :] < n)

        # Materialize a private LU copy; the input tensor remains unchanged.
        src = src_ptr + rows[:, None] * n + cols[None, :]
        aa = lu_ptr + rows[:, None] * n + cols[None, :]
        vv = tl.load(src, mask=full_mask, other=0.0)
        tl.store(aa, vv, mask=full_mask)

        rhs = tl.load(b_ptr + rows)
        for k in range(n):
            active = rows >= k
            col = tl.load(lu_ptr + rows * n + k, mask=active, other=0.0)
            pivot = tl.argmax(tl.abs(col), axis=0)

            # Swap the pivot row into position k, including the RHS.  The
            # vectorized row exchange is independent across columns.
            rk = tl.load(lu_ptr + k * n + cols)
            rp = tl.load(lu_ptr + pivot * n + cols)
            # Previous L columns must be exchanged as well; otherwise later
            # rows would no longer represent the same row permutations.
            sw = cols < n
            tl.store(lu_ptr + k * n + cols, tl.where(sw, rp, rk))
            tl.store(lu_ptr + pivot * n + cols, tl.where(sw, rk, rp))
            bk = tl.sum(tl.where(rows == k, rhs, 0.0), axis=0)
            bp = tl.sum(tl.where(rows == pivot, rhs, 0.0), axis=0)
            rhs = tl.where(rows == k, bp, tl.where(rows == pivot, bk, rhs))

            pv = tl.load(lu_ptr + k * n + k)
            below = rows > k
            factors = tl.load(lu_ptr + rows * n + k, mask=below, other=0.0) / pv
            # Store L's multiplier before updating the trailing U block.
            tl.store(lu_ptr + rows * n + k, factors, mask=below)
            trailing = (rows[:, None] > k) & (cols[None, :] > k)
            left = factors[:, None]
            prow = tl.load(lu_ptr + k * n + cols)
            old = tl.load(lu_ptr + rows[:, None] * n + cols[None, :], mask=full_mask, other=0.0)
            tl.store(lu_ptr + rows[:, None] * n + cols[None, :],
                    tl.where(trailing, old - left * prow[None, :], old), mask=full_mask)

        # Forward substitution with the unit lower triangular factor.
        y = rhs
        for i in range(n):
            li = tl.load(lu_ptr + i * n + cols)
            yi = tl.sum(tl.where(cols < i, li * y, 0.0), axis=0)
            val = tl.sum(tl.where(cols == i, rhs, 0.0), axis=0) - yi
            y = tl.where(rows == i, val, y)

        # Back substitution in U.  The already computed suffix is held in x.
        x = y
        for i in range(n - 1, -1, -1):
            ui = tl.load(lu_ptr + i * n + cols)
            suffix = tl.sum(tl.where(cols > i, ui * x, 0.0), axis=0)
            diag = tl.sum(tl.where(cols == i, ui, 0.0), axis=0)
            val = (tl.sum(tl.where(cols == i, y, 0.0), axis=0) - suffix) / diag
            x = tl.where(rows == i, val, x)
        tl.store(out_ptr + rows, x)


def build(context):
    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        lu = torch.empty_like(A)
        out = torch.empty_like(b)
        # The kernel copies A into lu before modifying it; lu is an explicit
        # cross-boundary buffer and therefore does not alias either input.
        _fused_lu_solve[(1,)](A, lu, b, out, n=256)
        return out

    return fused_lu_solve
