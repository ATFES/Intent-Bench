from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _symmetric_mm_abs_partial_kernel(
    a_ptr,
    c_ptr,
    partial_ptr,
    alpha,
    beta,
    m_rows,
    k_cols,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    num_col_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    if tile_m > tile_n:
        return

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m_rows
    col_mask = cols < k_cols
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, k_cols, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        k_mask = ks < k_cols
        lhs = tl.load(
            a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs_rows = tl.load(
            a_ptr + cols[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            lhs,
            tl.trans(rhs_rows),
            accumulator,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    tile_mask = row_mask[:, None] & col_mask[None, :]
    upper_c = tl.load(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        mask=tile_mask,
        other=0.0,
    )
    upper_d = alpha * accumulator + beta * upper_c
    if tile_m < tile_n:
        lower_c = tl.load(
            c_ptr + cols[:, None] * stride_cm + rows[None, :] * stride_cn,
            mask=tile_mask,
            other=0.0,
        )
        lower_d = alpha * tl.trans(accumulator) + beta * lower_c
    partial = tl.sum(tl.abs(upper_d), dtype=tl.float32)
    if tile_m < tile_n:
        partial += tl.sum(tl.abs(lower_d), dtype=tl.float32)
    partial_index = tile_m * num_col_tiles + tile_n
    tl.store(partial_ptr + partial_index, partial)


@triton.jit
def _reduce_abs_partials_kernel(
    partial_ptr,
    output_ptr,
    num_partials,
    num_col_tiles,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    in_range = offsets < num_partials
    tile_m = offsets // num_col_tiles
    tile_n = offsets % num_col_tiles
    active = in_range & (tile_m <= tile_n)
    partials = tl.load(partial_ptr + offsets, mask=active, other=0.0)
    total = tl.sum(partials, dtype=tl.float32)
    tl.store(output_ptr, total)


def build(context):
    def symmetric_mm_and_abs_sum(
        A: torch.Tensor,
        C: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.5,
    ) -> torch.Tensor:
        m_rows, k_cols = A.shape
        block_size = 128
        num_row_tiles = triton.cdiv(m_rows, block_size)
        num_col_tiles = triton.cdiv(k_cols, block_size)
        partials = torch.empty(
            (num_row_tiles * num_col_tiles,),
            device=A.device,
            dtype=torch.float32,
        )
        output = torch.empty((), device=A.device, dtype=torch.float32)

        _symmetric_mm_abs_partial_kernel[(num_row_tiles, num_col_tiles)](
            A,
            C,
            partials,
            alpha,
            beta,
            m_rows,
            k_cols,
            A.stride(0),
            A.stride(1),
            C.stride(0),
            C.stride(1),
            num_col_tiles,
            BLOCK_M=block_size,
            BLOCK_N=block_size,
            BLOCK_K=32,
            num_warps=8,
        )

        reduction_size = num_row_tiles * num_col_tiles
        _reduce_abs_partials_kernel[(1,)](
            partials,
            output,
            reduction_size,
            num_col_tiles,
            BLOCK_SIZE=triton.next_power_of_2(reduction_size),
            num_warps=4,
        )
        return output

    return symmetric_mm_and_abs_sum
