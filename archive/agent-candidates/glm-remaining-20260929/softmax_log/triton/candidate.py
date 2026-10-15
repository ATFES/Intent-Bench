from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_log_rows(
    input_ptr,
    output_ptr,
    row_stride,
    element_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    offsets = row.to(tl.int64) * row_stride + columns.to(tl.int64) * element_stride
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    invalid = (
        (values != values)
        | (values < 0.0)
        | ((values == 0.0) & (values.to(tl.int32, bitcast=True) < 0))
    )
    invalid = invalid & mask
    log_values = tl.where(
        values > 0.0,
        tl.log(values),
        tl.where(values == 0.0, float("-inf"), float("nan")),
    )
    log_values = tl.where(mask, log_values, float("-inf"))

    has_invalid = tl.max(invalid.to(tl.int32), axis=0) > 0
    maximum = tl.max(log_values, axis=0)
    maximum = tl.where(has_invalid, float("nan"), maximum)
    shifted = log_values - maximum
    weights = tl.exp(shifted)
    weights = tl.where(mask, weights, 0.0)
    denominator = tl.sum(weights, axis=0)
    probabilities = tl.fdiv(weights, denominator, ieee_rounding=True)

    tl.store(output_ptr + offsets, probabilities, mask=mask)


def build(context):
    def softmax_log(input, dim=-1, dtype=None):
        if not input.is_cuda:
            raise ValueError("softmax_log requires a CUDA input")
        if dtype is not None:
            raise ValueError("this fixed softmax_log profile requires dtype=None")
        if input.dim() == 0:
            raise ValueError("softmax_log requires at least one dimension")

        normalized_dim = dim if dim >= 0 else input.dim() + dim
        if normalized_dim != input.dim() - 1:
            raise ValueError("this fixed softmax_log profile requires the final dim")
        if input.stride(-1) != 1:
            raise ValueError("the final input dimension must be contiguous")

        n_cols = input.shape[-1]
        block_size = triton.next_power_of_2(n_cols)
        n_rows = input.numel() // n_cols
        output = torch.empty_like(input)

        _softmax_log_rows[(n_rows,)](
            input,
            output,
            input.stride(0) if input.dim() > 1 else n_cols,
            input.stride(-1),
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        return output

    return softmax_log
