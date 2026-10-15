from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    matrix_ptr,
    vector_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_rows,
    n_cols,
    matrix_row_stride,
    matrix_col_stride,
    vector_stride,
    output_stride,
    OTHER_IS_PTR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < n_rows
    accumulator = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

    for col_block in range(0, n_cols, BLOCK_COLS):
        cols = col_block + tl.arange(0, BLOCK_COLS)
        col_mask = cols < n_cols
        vector = tl.load(
            vector_ptr + cols * vector_stride,
            mask=col_mask,
            other=0.0,
        )
        offsets = (
            rows[:, None] * matrix_row_stride
            + cols[None, :] * matrix_col_stride
        )
        tile_mask = row_mask[:, None] & col_mask[None, :]
        matrix_tile = tl.load(
            matrix_ptr + offsets,
            mask=tile_mask,
            other=0.0,
        )
        accumulator += matrix_tile * vector[None, :]

    dot_product = tl.sum(accumulator, axis=1)
    sigmoid_value = 1.0 / (1.0 + tl.exp(-dot_product))

    if OTHER_IS_PTR:
        other_value = tl.load(other_ptr) * alpha
    else:
        other_value = other_ptr * alpha

    result = sigmoid_value - other_value
    tl.store(output_ptr + rows * output_stride, result, mask=row_mask)


def build(context):
    def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
        if input.device.type != "cuda" or vec.device.type != "cuda":
            raise ValueError("input and vec must be CUDA tensors")
        if input.ndim != 2 or vec.ndim != 1:
            raise ValueError("input must be 2-D and vec must be 1-D")
        if input.shape[1] != vec.shape[0]:
            raise ValueError("input and vec have incompatible matmul shapes")
        if input.dtype != torch.float32 or vec.dtype != torch.float32:
            raise ValueError("input and vec must have dtype float32")
        if input.device != vec.device:
            raise ValueError("input and vec must be on the same device")

        n_rows, n_cols = input.shape
        if out is None:
            output = torch.empty(n_rows, device=input.device, dtype=torch.float32)
        else:
            if out.device != input.device or out.dtype != torch.float32:
                raise ValueError("out must be a CUDA float32 tensor")
            if out.shape != (n_rows,):
                raise ValueError("out has an incompatible shape")
            output = out

        other_is_ptr = torch.is_tensor(other)
        if other_is_ptr:
            if other.device != input.device or other.dtype != torch.float32:
                raise ValueError("other must be a CUDA float32 tensor")
            if other.numel() != 1:
                raise ValueError("only scalar-valued other tensors are supported")
            other_arg = other
        else:
            other_arg = other

        grid = (triton.cdiv(n_rows, 4),)
        _fused_mv_sigmoid_sub_kernel[grid](
            input,
            vec,
            other_arg,
            output,
            alpha,
            n_rows,
            n_cols,
            input.stride(0),
            input.stride(1),
            vec.stride(0),
            output.stride(0),
            OTHER_IS_PTR=other_is_ptr,
            BLOCK_ROWS=4,
            BLOCK_COLS=128,
            num_warps=4,
        )
        return output

    return fused_mv_sigmoid_sub
