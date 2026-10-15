import torch
import triton
import triton.language as tl


@triton.jit
def _index_select_eq_kernel(
    input_ptr,
    index_ptr,
    other_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    input_col_stride,
    index_stride,
    output_row_stride,
    output_col_stride,
    other_row_stride,
    other_col_stride,
    scalar_other,
    DIM: tl.constexpr,
    OTHER_IS_SCALAR: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row_tile = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col_tile = tl.program_id(1) * COLS + tl.arange(0, COLS)
    row_mask = row_tile < n_rows
    col_mask = col_tile < n_cols
    tile_mask = row_mask[:, None] & col_mask[None, :]

    if DIM == 0:
        index_offsets = row_tile * index_stride
        selected_rows = tl.load(index_ptr + index_offsets, mask=row_mask, other=0)
        selected_cols = col_tile
    else:
        index_offsets = col_tile * index_stride
        selected_rows = row_tile
        selected_cols = tl.load(index_ptr + index_offsets, mask=col_mask, other=0)

    input_offsets = (
        selected_rows[:, None] * input_row_stride
        + selected_cols[None, :] * input_col_stride
    )
    selected = tl.load(input_ptr + input_offsets, mask=tile_mask, other=0.0)

    if OTHER_IS_SCALAR:
        equal = selected == scalar_other
    else:
        other_offsets = (
            row_tile[:, None] * other_row_stride
            + col_tile[None, :] * other_col_stride
        )
        other_values = tl.load(other_ptr + other_offsets, mask=tile_mask, other=0.0)
        equal = selected == other_values

    output_offsets = (
        row_tile[:, None] * output_row_stride
        + col_tile[None, :] * output_col_stride
    )
    tl.store(output_ptr + output_offsets, equal, mask=tile_mask)


def _broadcast_strides(selected_shape, other):
    if other.ndim > 2:
        leading_size = 1
        for size in other.shape[:-2]:
            leading_size *= size
            if leading_size != 1:
                raise ValueError("other is not broadcastable to the selected shape")

    result = []
    for axis, size in enumerate(selected_shape):
        other_axis = axis + 2 - min(other.ndim, 2)
        if other_axis < 0:
            stride = 0
        else:
            other_size = other.shape[other_axis]
            if other_size == 1:
                stride = 0
            else:
                if other_size != size:
                    raise ValueError("other is not broadcastable to the selected shape")
                stride = other.stride(other_axis)
        result.append(stride)
    return result


def build(context):
    def fused_index_select_eq(input, dim, index, other, *, out=None):
        if input.ndim != 2:
            raise ValueError("input must be 2-dimensional")
        if index.ndim != 1:
            raise ValueError("index must be 1-dimensional")
        if index.dtype not in (torch.int32, torch.int64):
            raise ValueError("index must be an IntTensor or LongTensor")

        normalized_dim = dim if dim >= 0 else dim + input.ndim
        if normalized_dim not in (0, 1):
            raise ValueError("Dimension out of range (expected to be in range of [-2, 1])")
        elif normalized_dim == 0:
            selected_shape = (index.shape[0], input.shape[1])
        else:
            selected_shape = (input.shape[0], index.shape[0])

        if out is None:
            output = torch.empty(
                selected_shape, dtype=torch.bool, device=input.device
            )
        else:
            if out.shape != selected_shape:
                raise ValueError("out has an incompatible shape")
            if out.dtype != torch.bool:
                raise ValueError("out must have dtype torch.bool")
            output = out

        other_is_scalar = not torch.is_tensor(other)
        if other_is_scalar:
            scalar_other = float(other)
            other_ptr = input
            other_row_stride = 0
            other_col_stride = 0
        else:
            scalar_other = 0.0
            other_row_stride, other_col_stride = _broadcast_strides(
                selected_shape, other
            )
            other_ptr = other

        row_tiles = triton.cdiv(selected_shape[0], 8)
        col_tiles = triton.cdiv(selected_shape[1], 128)
        _index_select_eq_kernel[(row_tiles, col_tiles)](
            input,
            index,
            other_ptr,
            output,
            selected_shape[0],
            selected_shape[1],
            input.stride(0),
            input.stride(1),
            index.stride(0),
            output.stride(0),
            output.stride(1),
            other_row_stride,
            other_col_stride,
            scalar_other,
            DIM=normalized_dim,
            OTHER_IS_SCALAR=other_is_scalar,
            ROWS=8,
            COLS=128,
        )
        return output

    return fused_index_select_eq
