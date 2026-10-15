from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_index_select_eq_kernel(
    x_ptr,
    index_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    other: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)

    idx = tl.load(index_ptr + row).to(tl.int32)
    x_ptrs = x_ptr + idx * n_cols + cols
    out_ptrs = out_ptr + row * n_cols + cols

    other_val = tl.full((BLOCK_COLS,), other, tl.float32)

    if BLOCK_COLS == n_cols:
        x = tl.load(x_ptrs)
        tl.store(out_ptrs, x == other_val)
    else:
        mask = cols < n_cols
        x = tl.load(x_ptrs, mask=mask)
        tl.store(out_ptrs, x == other_val, mask=mask)


def fused_index_select_eq(input, dim, index, other, *, out=None):
    if dim < 0:
        dim += input.dim()

    # The fixed invocation profile is a 2D contiguous input indexed on dim 0.
    if dim != 0 or input.dim() != 2:
        raise NotImplementedError(
            "fused_index_select_eq currently supports dim=0 on 2D inputs"
        )

    n_rows = index.numel()
    n_cols = input.shape[1]

    if out is None:
        out = torch.empty((n_rows, n_cols), dtype=torch.bool, device=input.device)

    BLOCK_COLS = triton.next_power_of_2(n_cols)
    _fused_index_select_eq_kernel[(n_rows,)](
        input,
        index,
        out,
        n_cols=n_cols,
        other=other,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=1,
    )

    return out


def build(context):
    return fused_index_select_eq
