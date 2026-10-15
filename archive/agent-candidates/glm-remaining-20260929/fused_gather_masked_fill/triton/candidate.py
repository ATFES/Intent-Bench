from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _fused_gather_masked_fill_kernel(
    input_ptr,
    index_ptr,
    mask_ptr,
    output_ptr,
    fill_value,
    rows,
    cols,
    input_stride_0,
    input_stride_1,
    index_stride_0,
    index_stride_1,
    mask_stride_0,
    mask_stride_1,
    output_stride_0,
    output_stride_1,
    gather_dim: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    column_offset = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    valid = column_offset < cols

    index_offset = row * index_stride_0 + column_offset * index_stride_1
    indices = tl.load(index_ptr + index_offset, mask=valid, other=0)
    mask_offset = row * mask_stride_0 + column_offset * mask_stride_1
    selected = tl.load(mask_ptr + mask_offset, mask=valid, other=0)

    if gather_dim == 0:
        input_offset = indices * input_stride_0 + column_offset * input_stride_1
    else:
        input_offset = row * input_stride_0 + indices * input_stride_1
    gathered = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
    result = tl.where(selected, fill_value, gathered)

    output_offset = row * output_stride_0 + column_offset * output_stride_1
    tl.store(output_ptr + output_offset, result, mask=valid)


def build(context):
    def fused_gather_masked_fill(
        input,
        dim,
        index,
        mask,
        value,
        *,
        sparse_grad=False,
        out=None,
    ):
        if input.ndim != index.ndim:
            raise ValueError("input and index must have the same number of dimensions")
        if input.ndim != 2:
            raise ValueError("the fixed Triton implementation requires a 2-D input")
        if dim not in (0, 1):
            raise ValueError("dim must be 0 or 1")
        if index.shape[1 - dim] > input.shape[1 - dim]:
            raise ValueError("index size must not exceed input outside the gather dimension")

        rows, cols = index.shape
        if out is None:
            output = input.new_empty((rows, cols))
        else:
            output = out

        if mask.ndim == 0:
            mask_stride_0 = 0
            mask_stride_1 = 0
        else:
            if mask.ndim > 2 or any(size != 1 for size in mask.shape[:-2]):
                raise ValueError("mask is not broadcastable to the output")
            mask_shape = (1,) * (2 - mask.ndim) + mask.shape
            mask_strides = (0,) * (2 - mask.ndim) + mask.stride()
            mask_stride_0 = mask_strides[0] if mask_shape[0] != 1 else 0
            mask_stride_1 = mask_strides[1] if mask_shape[1] != 1 else 0
            if mask_shape[0] not in (1, rows) or mask_shape[1] not in (1, cols):
                raise ValueError("mask is not broadcastable to the output")

        block_cols = 256
        _fused_gather_masked_fill_kernel[(rows, triton.cdiv(cols, block_cols))](
            input,
            index,
            mask,
            output,
            value,
            rows,
            cols,
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            mask_stride_0,
            mask_stride_1,
            output.stride(0),
            output.stride(1),
            gather_dim=dim,
            BLOCK_COLS=block_cols,
        )
        return output

    return fused_gather_masked_fill
