from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_pair_combine(left_value, left_index, right_value, right_index):
    left_is_nan = left_value != left_value
    right_is_nan = right_value != right_value

    left_beats_finite = left_is_nan & (right_is_nan == 0)
    left_beats_nan = left_is_nan & right_is_nan & (left_index < right_index)
    left_beats_value = (
        (left_is_nan == 0)
        & (right_is_nan == 0)
        & (
            (left_value > right_value)
            | ((left_value == right_value) & (left_index < right_index))
        )
    )
    left_wins = left_beats_finite | left_beats_nan | left_beats_value

    return tl.where(left_wins, left_value, right_value), tl.where(
        left_wins, left_index, right_index
    )


@triton.jit
def _partial_argmax_kernel(
    input_ptr,
    value_ptr,
    index_ptr,
    input_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(axis=0).to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    indices = tl.where(mask, offsets, 2147483647)

    value, index = tl.reduce(
        (values, indices), axis=0, combine_fn=_argmax_pair_combine
    )
    tl.store(value_ptr + tl.program_id(axis=0), value)
    tl.store(index_ptr + tl.program_id(axis=0), index)


@triton.jit
def _final_argmax_kernel(
    value_ptr,
    index_ptr,
    output_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials
    values = tl.load(value_ptr + offsets, mask=mask, other=0.0)
    indices = tl.where(mask, tl.load(index_ptr + offsets, mask=mask, other=0), 2147483647)

    _, index = tl.reduce((values, indices), axis=0, combine_fn=_argmax_pair_combine)
    tl.store(output_ptr, index)


def build(context):
    def argmax(input_tensor, dim, keepdim=False):
        original_dim = dim
        if dim is None:
            if not input_tensor.is_contiguous():
                raise NotImplementedError("flattening a non-contiguous tensor")
            input_tensor = input_tensor.view(-1)
            dim = 0

        if input_tensor.ndim != 1:
            raise NotImplementedError("only the supplied one-dimensional profile")
        if input_tensor.dtype != torch.float16 or input_tensor.device.type != "cuda":
            raise NotImplementedError("only the supplied fp16 CUDA profile")

        dim = dim % input_tensor.ndim
        if dim != 0:
            raise NotImplementedError("only the supplied reduction dimension")

        n_elements = input_tensor.numel()
        if n_elements == 0:
            raise RuntimeError("Expected reduction dim 0 to have non-zero size")

        block_size = 2048
        num_partials = triton.cdiv(n_elements, block_size)
        partial_values = torch.empty(
            (num_partials,), dtype=torch.float16, device=input_tensor.device
        )
        partial_indices = torch.empty(
            (num_partials,), dtype=torch.int64, device=input_tensor.device
        )
        output = torch.empty((), dtype=torch.int64, device=input_tensor.device)

        _partial_argmax_kernel[(num_partials,)](
            input_tensor,
            partial_values,
            partial_indices,
            input_tensor.stride(0),
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        _final_argmax_kernel[(1,)](
            partial_values,
            partial_indices,
            output,
            num_partials,
            BLOCK_SIZE=triton.next_power_of_2(num_partials),
            num_warps=4,
        )

        if keepdim:
            return output.view((1,) if original_dim is not None else (1,))
        return output.view(())

    return argmax
