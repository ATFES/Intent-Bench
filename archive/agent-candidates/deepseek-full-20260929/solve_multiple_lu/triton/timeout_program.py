from __future__ import annotations

import math
from typing import Any, Callable

import torch
import triton
import triton.language as tl


@triton.jit
def _lu_solve_kernel(
    A,
    B,
    LU,
    X,
    n,
    k,
    lda,
    ldb,
    ldlu,
    ldx,
    PIVOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    batch = tl.program_id(0)
    a_base = batch * n * n
    b_base = batch * n * k
    lu_base = batch * n * n
    x_base = batch * n * k

    rm = tl.arange(0, BLOCK_M)[:, None]
    cn = tl.arange(0, BLOCK_N)
    rn = cn[None, :]
    pr = tl.arange(0, BLOCK_P)

    # Copy A -> LU.
    for r0 in tl.range(0, n, BLOCK_M):
        r = r0 + rm
        for c0 in tl.range(0, n, BLOCK_N):
            c = c0 + rn
            mask = (r < n) & (c < n)
            v = tl.load(A + a_base + r * lda + c, mask=mask, other=0.0)
            tl.store(LU + lu_base + r * ldlu + c, v, mask=mask)

    # Copy B -> X.  X serves as the permuted right-hand side and final result.
    for r0 in tl.range(0, n, BLOCK_M):
        r = r0 + rm
        for c0 in tl.range(0, k, BLOCK_N):
            c = c0 + rn
            mask = (r < n) & (c < k)
            v = tl.load(B + b_base + r * ldb + c, mask=mask, other=0.0)
            tl.store(X + x_base + r * ldx + c, v, mask=mask)

    # In-place Doolittle LU decomposition.
    for kk in tl.range(0, n):
        if PIVOT:
            best_val = tl.full((), float("-inf"), tl.float32)
            best_idx = tl.full((), -1, tl.int32)
            for r0 in tl.range(0, n, BLOCK_P):
                rr = r0 + pr
                valid = (rr < n) & (rr >= kk)
                v = tl.load(LU + lu_base + rr * ldlu + kk, mask=valid, other=0.0)
                av = tl.where(valid, tl.abs(v), float("-inf"))
                local_max, local_idx = tl.max(
                    av,
                    axis=0,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                local_idx = local_idx + r0
                better = local_max > best_val
                best_val = tl.where(better, local_max, best_val)
                best_idx = tl.where(better, local_idx, best_idx)
            piv = best_idx
        else:
            piv = kk

        if piv != kk:
            # Swap the two rows of the working LU matrix.
            for c0 in tl.range(0, n, BLOCK_N):
                c = c0 + cn
                mask = c < n
                p0 = LU + lu_base + kk * ldlu + c
                p1 = LU + lu_base + piv * ldlu + c
                v0 = tl.load(p0, mask=mask, other=0.0)
                v1 = tl.load(p1, mask=mask, other=0.0)
                tl.store(p0, v1, mask=mask)
                tl.store(p1, v0, mask=mask)

            # Apply the same permutation to the right-hand sides.
            for c0 in tl.range(0, k, BLOCK_N):
                c = c0 + cn
                mask = c < k
                p0 = X + x_base + kk * ldx + c
                p1 = X + x_base + piv * ldx + c
                v0 = tl.load(p0, mask=mask, other=0.0)
                v1 = tl.load(p1, mask=mask, other=0.0)
                tl.store(p0, v1, mask=mask)
                tl.store(p1, v0, mask=mask)

        diag = tl.load(LU + lu_base + kk * ldlu + kk)

        # Eliminate all rows below the current pivot row.
        for r0 in tl.range(kk + 1, n, BLOCK_M):
            r = r0 + rm
            row_mask = (r < n) & (r > kk)
            factors = tl.load(LU + lu_base + r * ldlu + kk, mask=row_mask, other=0.0) / diag
            tl.store(LU + lu_base + r * ldlu + kk, factors, mask=row_mask)

            for c0 in tl.range(kk + 1, n, BLOCK_N):
                c = c0 + rn
                col_mask = (c < n) & (c > kk)
                mask = row_mask & col_mask
                u = tl.load(LU + lu_base + kk * ldlu + c, mask=col_mask, other=0.0)
                v = tl.load(LU + lu_base + r * ldlu + c, mask=mask, other=0.0)
                tl.store(LU + lu_base + r * ldlu + c, v - factors * u, mask=mask)

    # Forward substitution: L y = (P^T B).
    for i in tl.range(0, n):
        for c0 in tl.range(0, k, BLOCK_N):
            c = c0 + cn
            mask = c < k
            acc = tl.load(X + x_base + i * ldx + c, mask=mask, other=0.0)
            for j in tl.range(0, i):
                lij = tl.load(LU + lu_base + i * ldlu + j)
                xj = tl.load(X + x_base + j * ldx + c, mask=mask, other=0.0)
                acc = acc - lij * xj
            tl.store(X + x_base + i * ldx + c, acc, mask=mask)

    # Back substitution: U x = y.
    for ii in tl.range(0, n):
        i = n - 1 - ii
        for c0 in tl.range(0, k, BLOCK_N):
            c = c0 + cn
            mask = c < k
            acc = tl.load(X + x_base + i * ldx + c, mask=mask, other=0.0)
            for j in tl.range(i + 1, n):
                uij = tl.load(LU + lu_base + i * ldlu + j)
                xj = tl.load(X + x_base + j * ldx + c, mask=mask, other=0.0)
                acc = acc - uij * xj
            diag = tl.load(LU + lu_base + i * ldlu + i)
            tl.store(X + x_base + i * ldx + c, acc / diag, mask=mask)


def build(context: Any) -> Callable[..., torch.Tensor]:
    def solve_multiple_lu(A: torch.Tensor, Bs: torch.Tensor, *, pivot: bool = True, out: torch.Tensor | None = None) -> torch.Tensor:
        if A.dtype != torch.float32 or Bs.dtype != torch.float32:
            raise NotImplementedError("solve_multiple_lu supports float32 tensors only")

        n = A.shape[-1]
        k = Bs.shape[-1]
        batch_shape = A.shape[:-2]
        num_batches = math.prod(batch_shape) if batch_shape else 1

        if not A.is_contiguous() or not Bs.is_contiguous():
            raise NotImplementedError("solve_multiple_lu requires contiguous A and Bs")

        A_flat = A.view(num_batches, n, n)
        B_flat = Bs.view(num_batches, n, k)
        LU = torch.empty((num_batches, n, n), device=A.device, dtype=A.dtype)

        if out is None:
            X = torch.empty((*batch_shape, n, k), device=A.device, dtype=A.dtype)
            X_flat = X.view(num_batches, n, k)
            ret = X
        else:
            if out.shape != (*batch_shape, n, k):
                raise ValueError("out has an incompatible shape")
            if out.dtype != A.dtype:
                raise ValueError("out has an incompatible dtype")
            if not out.is_contiguous():
                raise NotImplementedError("solve_multiple_lu requires contiguous out")
            X = out
            X_flat = X.view(num_batches, n, k)
            ret = out

        block_n = min(triton.next_power_of_2(max(n, k)), 1024)
        block_m = max(1, 1024 // block_n)
        block_p = min(triton.next_power_of_2(n), 1024)
        num_warps = max(1, (block_m * block_n) // 32)

        _lu_solve_kernel[(num_batches,)](
            A_flat,
            B_flat,
            LU,
            X_flat,
            n,
            k,
            A_flat.stride(1),
            B_flat.stride(1),
            LU.stride(1),
            X_flat.stride(1),
            PIVOT=pivot,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_P=block_p,
            num_warps=num_warps,
        )

        return ret

    return solve_multiple_lu
