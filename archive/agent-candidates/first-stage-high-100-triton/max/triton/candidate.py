import torch
import triton
import triton.language as tl


@triton.jit
def _max_index_combine(value_left, index_left, value_right, index_right):
    # NaNs propagate as they do for torch.max, and the first NaN wins.  For
    # ordinary values, retaining the left operand on equality gives the
    # required first-occurrence argmax.
    left_is_nan = value_left != value_left
    left_is_not_nan = value_left == value_left
    right_is_nan = value_right != value_right
    right_is_not_nan = value_right == value_right
    choose_left = (
        left_is_nan & (right_is_not_nan | (index_left < index_right))
    ) | (
        left_is_not_nan
        & right_is_not_nan
        & ((value_left > value_right)
           | ((value_left == value_right) & (index_left < index_right)))
    )
    value = tl.where(choose_left, value_left, value_right)
    index = tl.where(choose_left, index_left, index_right)
    return value, index


@triton.jit
def _max_index_stage1(
    input_ptr,
    partial_value_ptr,
    partial_index_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    indices = tl.where(mask, offsets, n_elements)
    value, index = tl.reduce(
        (values, indices), axis=0, combine_fn=_max_index_combine
    )

    tl.store(partial_value_ptr + program, value)
    tl.store(partial_index_ptr + program, index)


@triton.jit
def _max_index_stage2(
    partial_value_ptr,
    partial_index_ptr,
    value_out_ptr,
    index_out_ptr,
    n_partials,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial_value_ptr + offsets, mask=mask, other=-float("inf"))
    indices = tl.load(partial_index_ptr + offsets, mask=mask, other=n_partials)
    value, index = tl.reduce(
        (values, indices), axis=0, combine_fn=_max_index_combine
    )

    tl.store(value_out_ptr, value)
    tl.store(index_out_ptr, index.to(tl.int64))


def build(context):
    def wrapper(input, dim, keepdim=False, *, out=None):
        n_elements = input.numel()
        block = 4096
        n_partials = (n_elements + block - 1) // block

        partial_values = torch.empty(
            (n_partials,), dtype=input.dtype, device=input.device
        )
        partial_indices = torch.empty(
            (n_partials,), dtype=torch.int32, device=input.device
        )

        _max_index_stage1[(n_partials,)](
            input,
            partial_values,
            partial_indices,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )

        if out is None:
            values_out = torch.empty((), dtype=input.dtype, device=input.device)
            indices_out = torch.empty((), dtype=torch.int64, device=input.device)
        else:
            values_out, indices_out = out

        _max_index_stage2[(1,)](
            partial_values,
            partial_indices,
            values_out,
            indices_out,
            n_partials,
            BLOCK=256,
            num_warps=4,
        )
        return values_out, indices_out

    return wrapper
