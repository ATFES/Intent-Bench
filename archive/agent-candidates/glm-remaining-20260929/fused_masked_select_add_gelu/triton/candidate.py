from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


RESULT_NUMEL = 524236


@triton.jit
def _count_blocks(mask_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    count = tl.sum(selected.to(tl.int32), axis=0)
    tl.store(partial_ptr + tl.program_id(0), count)


@triton.jit
def _gelu_erf(value: tl.tensor):
    erf_value = tl.erf(value * 0.7071067811865476)
    return 0.5 * value * (1.0 + erf_value)


@triton.jit
def _gelu_tanh(value: tl.tensor):
    scaled = 0.7978845608028654 * (value + 0.044715 * value * value * value)
    exponential = tl.exp(2.0 * scaled)
    tanh_value = 1.0 - 2.0 / (exponential + 1.0)
    return 0.5 * value * (1.0 + tanh_value)


@triton.jit
def _select_add_gelu_scalar(
    input_ptr,
    mask_ptr,
    partial_ptr,
    output_ptr,
    n_elements,
    alpha,
    other,
    APPROXIMATE: tl.constexpr,
    N_PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_offsets = tl.arange(0, N_PROGRAMS)
    block_counts = tl.load(partial_ptr + program_offsets)
    inclusive_counts = tl.cumsum(block_counts, axis=0)
    block_starts = inclusive_counts - block_counts
    output_start = tl.sum(
        tl.where(program_offsets == program_id, block_starts, 0), axis=0
    )

    offsets = program_id * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    selected_int = selected.to(tl.int32)
    local_rank = tl.cumsum(selected_int, axis=0) - selected_int
    value = tl.load(input_ptr + offsets, mask=selected, other=0.0)
    shifted = value + alpha * other
    if APPROXIMATE:
        result = _gelu_tanh(shifted)
    else:
        result = _gelu_erf(shifted)
    tl.store(output_ptr + output_start + local_rank, result, mask=selected)


@triton.jit
def _select_add_gelu_tensor(
    input_ptr,
    mask_ptr,
    other_ptr,
    partial_ptr,
    output_ptr,
    n_elements,
    other_stride,
    output_stride,
    alpha,
    APPROXIMATE: tl.constexpr,
    N_PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_offsets = tl.arange(0, N_PROGRAMS)
    block_counts = tl.load(partial_ptr + program_offsets)
    inclusive_counts = tl.cumsum(block_counts, axis=0)
    block_starts = inclusive_counts - block_counts
    output_start = tl.sum(
        tl.where(program_offsets == program_id, block_starts, 0), axis=0
    )

    offsets = program_id * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    selected_int = selected.to(tl.int32)
    local_rank = tl.cumsum(selected_int, axis=0) - selected_int
    output_index = output_start + local_rank
    value = tl.load(input_ptr + offsets, mask=selected, other=0.0)
    other = tl.load(
        other_ptr + output_index * other_stride, mask=selected, other=0.0
    )
    shifted = value + alpha * other
    if APPROXIMATE:
        result = _gelu_tanh(shifted)
    else:
        result = _gelu_erf(shifted)
    tl.store(output_ptr + output_index * output_stride, result, mask=selected)


def build(context):
    del context

    def fused_masked_select_add_gelu(
        input,
        mask,
        other,
        *,
        alpha=1,
        approximate="none",
        out=None,
    ):
        if input.ndim != 1 or mask.ndim != 1:
            raise ValueError("this kernel requires one-dimensional input and mask")
        if input.shape != mask.shape or not input.is_contiguous() or not mask.is_contiguous():
            raise ValueError("input and mask must have the same contiguous shape")

        n_elements = input.numel()
        n_programs = triton.cdiv(n_elements, 2048)
        block_counts = torch.empty(
            (n_programs,), dtype=torch.int32, device=input.device
        )
        _count_blocks[(n_programs,)](
            mask,
            block_counts,
            n_elements,
            BLOCK=2048,
        )

        approximate_is_tanh = approximate == "tanh"
        if approximate != "none" and not approximate_is_tanh:
            raise ValueError("approximate must be 'none' or 'tanh'")

        if out is None:
            output = torch.empty(
                (RESULT_NUMEL,), dtype=torch.float32, device=input.device
            )
        else:
            if out.shape != (RESULT_NUMEL,) or out.dtype != torch.float32:
                raise ValueError("out must have the documented float32 result shape")
            output = out

        if output.is_contiguous():
            output_stride = 1
        else:
            if output.stride(1 if output.ndim > 1 else 0) != 1:
                raise ValueError("out must have a contiguous unit inner stride")
            output_stride = output.stride(0)

        if isinstance(other, torch.Tensor):
            expanded_other = other.expand((RESULT_NUMEL,))
            if not expanded_other.is_contiguous() and expanded_other.stride(0) != 0:
                raise ValueError("other must be broadcastable to the selected shape")
            _select_add_gelu_tensor[(n_programs,)](
                input,
                mask,
                expanded_other,
                block_counts,
                output,
                n_elements,
                expanded_other.stride(0),
                output_stride,
                alpha,
                APPROXIMATE=approximate_is_tanh,
                N_PROGRAMS=n_programs,
                BLOCK=2048,
            )
        else:
            _select_add_gelu_scalar[(n_programs,)](
                input,
                mask,
                block_counts,
                output,
                n_elements,
                alpha,
                other,
                APPROXIMATE=approximate_is_tanh,
                N_PROGRAMS=n_programs,
                BLOCK=2048,
            )

        return output

    return fused_masked_select_add_gelu
