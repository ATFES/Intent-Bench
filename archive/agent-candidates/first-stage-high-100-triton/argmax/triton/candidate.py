import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_partials(
    input_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))

    # PyTorch selects the first index for equal values, and treats a NaN as
    # winning over ordinary values.  Keep the NaN path explicit because a
    # NaN cannot be found with an equality comparison.
    nan_index = tl.min(
        tl.where(mask & (values != values), offsets, n_elements), axis=0
    )
    max_value = tl.max(values, axis=0)
    value_index = tl.min(
        tl.where(mask & (values == max_value), offsets, n_elements), axis=0
    )
    has_nan = nan_index < n_elements
    selected_index = tl.where(has_nan, nan_index, value_index)

    safe_nan_index = tl.minimum(nan_index, n_elements - 1)
    nan_value = tl.load(input_ptr + safe_nan_index)
    selected_value = tl.where(has_nan, nan_value, max_value)

    tl.store(partial_values_ptr + pid, selected_value)
    tl.store(partial_indices_ptr + pid, selected_index)


@triton.jit
def _argmax_finish(
    partial_values_ptr,
    partial_indices_ptr,
    output_ptr,
    n_partials,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials

    values = tl.load(partial_values_ptr + offsets, mask=mask, other=-float("inf"))
    indices = tl.load(partial_indices_ptr + offsets, mask=mask, other=0)

    nan_index = tl.min(
        tl.where(mask & (values != values), indices, n_elements), axis=0
    )
    max_value = tl.max(values, axis=0)
    value_index = tl.min(
        tl.where(mask & (values == max_value), indices, n_elements), axis=0
    )
    has_nan = nan_index < n_elements
    selected_index = tl.where(has_nan, nan_index, value_index)

    tl.store(output_ptr, selected_index.to(tl.int64))


def build(context):
    def wrapper(input, dim, keepdim=False):
        n_elements = input.numel()
        block = 1024
        n_partials = triton.cdiv(n_elements, block)

        partial_values = torch.empty(
            (n_partials,), dtype=input.dtype, device=input.device
        )
        partial_indices = torch.empty(
            (n_partials,), dtype=torch.int32, device=input.device
        )
        output_shape = (1,) if keepdim else ()
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

        _argmax_partials[(n_partials,)](
            input,
            partial_values,
            partial_indices,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )
        _argmax_finish[(1,)](
            partial_values,
            partial_indices,
            output,
            n_partials,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )
        return output

    return wrapper
