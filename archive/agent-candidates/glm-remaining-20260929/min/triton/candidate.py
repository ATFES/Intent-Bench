from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _argmin_combine(left_value, left_index, right_value, right_index):
    left_is_nan = left_value != left_value
    right_is_nan = right_value != right_value
    right_wins = right_is_nan & (~left_is_nan)
    right_wins |= (
        (~left_is_nan)
        & (~right_is_nan)
        & (
            (right_value < left_value)
            | ((right_value == left_value) & (right_index < left_index))
        )
    )
    return tl.where(right_wins, right_value, left_value), tl.where(
        right_wins, right_index, left_index
    )


@triton.jit
def _stage1_argmin_kernel(
    input_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    num_rows,
    num_cols,
    input_row_stride,
    input_col_stride,
    partial_group_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    column_block = tl.program_id(0)
    row_block = tl.program_id(1)
    columns = column_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column_mask = columns < num_cols
    row_mask = rows < num_rows
    input_ptrs = (
        input_ptr
        + rows[:, None] * input_row_stride
        + columns[None, :] * input_col_stride
    )
    values = tl.load(
        input_ptrs,
        mask=row_mask[:, None] & column_mask[None, :],
        other=float("inf"),
    )
    row_indices = tl.broadcast_to(rows[:, None], (BLOCK_ROWS, BLOCK_COLS))
    tile_values, tile_indices = tl.reduce(
        (values, row_indices), axis=0, combine_fn=_argmin_combine
    )
    partial_offsets = row_block * partial_group_stride + columns
    tl.store(partial_values_ptr + partial_offsets, tile_values, mask=column_mask)
    tl.store(partial_indices_ptr + partial_offsets, tile_indices, mask=column_mask)


@triton.jit
def _stage2_argmin_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_ptr,
    index_ptr,
    num_groups,
    num_cols,
    partial_group_stride,
    PARTIAL_GROUPS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    columns = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    groups = tl.arange(0, PARTIAL_GROUPS)
    column_mask = columns < num_cols
    group_mask = groups < num_groups
    partial_ptrs = (
        partial_values_ptr
        + groups[:, None] * partial_group_stride
        + columns[None, :]
    )
    values = tl.load(
        partial_ptrs,
        mask=group_mask[:, None] & column_mask[None, :],
        other=float("inf"),
    )
    index_ptrs = (
        partial_indices_ptr
        + groups[:, None] * partial_group_stride
        + columns[None, :]
    )
    indices = tl.load(
        index_ptrs,
        mask=group_mask[:, None] & column_mask[None, :],
        other=0,
    )
    output_values, output_indices = tl.reduce(
        (values, indices), axis=0, combine_fn=_argmin_combine
    )
    tl.store(output_ptr + columns, output_values, mask=column_mask)
    tl.store(index_ptr + columns, output_indices.to(tl.int64), mask=column_mask)


def build(context):
    def min(input_tensor, dim, keepdim=False):
        if dim != 0:
            raise ValueError("this kernel supports the supplied dim=0 profile")

        num_rows, num_cols = input_tensor.shape
        num_groups = triton.cdiv(num_rows, BLOCK_ROWS)
        num_col_blocks = triton.cdiv(num_cols, BLOCK_COLS)
        values = torch.empty(
            (1, num_cols) if keepdim else (num_cols,),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        indices = torch.empty(
            (1, num_cols) if keepdim else (num_cols,),
            dtype=torch.int64,
            device=input_tensor.device,
        )
        partial_values = torch.empty(
            (num_groups, num_cols),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        partial_indices = torch.empty(
            (num_groups, num_cols),
            dtype=torch.int32,
            device=input_tensor.device,
        )
        _stage1_argmin_kernel[(num_col_blocks, num_groups)](
            input_tensor,
            partial_values,
            partial_indices,
            num_rows,
            num_cols,
            input_tensor.stride(0),
            input_tensor.stride(1),
            num_cols,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=NUM_WARPS,
        )
        _stage2_argmin_kernel[(num_col_blocks,)](
            partial_values,
            partial_indices,
            values,
            indices,
            num_groups,
            num_cols,
            num_cols,
            PARTIAL_GROUPS=PARTIAL_GROUPS,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=NUM_WARPS,
        )
        return values, indices

    return min


PARTIAL_GROUPS = 16
BLOCK_ROWS = 64
BLOCK_COLS = 64
NUM_WARPS = 4
