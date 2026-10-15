from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    logits_ptr,
    targets_ptr,
    probabilities_ptr,
    row_losses_ptr,
    n_cols,
    logits_row_stride,
    output_row_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    column_offsets = tl.arange(0, BLOCK_SIZE)
    column_mask = column_offsets < n_cols
    row_offset = row * logits_row_stride
    logits = tl.load(
        logits_ptr + row_offset + column_offsets,
        mask=column_mask,
        other=float("-inf"),
    )

    maximum = tl.max(logits, axis=0)
    shifted = logits - maximum
    exponentials = tl.exp(shifted)
    exponential_sum = tl.sum(exponentials, axis=0)
    probabilities = exponentials / exponential_sum

    probabilities = tl.where(column_mask, probabilities, 0.0)
    mean = tl.sum(probabilities, axis=0) / n_cols
    centered = tl.where(column_mask, probabilities - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    inverse_stddev = 1.0 / tl.sqrt(variance + eps)
    normalized = centered * inverse_stddev

    output_offset = row * output_row_stride
    tl.store(
        probabilities_ptr + output_offset + column_offsets,
        normalized,
        mask=column_mask,
    )

    target = tl.load(targets_ptr + row)
    target_logit = tl.sum(
        tl.where(column_offsets == target, shifted, 0.0),
        axis=0,
    )
    loss = maximum + tl.log(exponential_sum) - target_logit
    tl.store(row_losses_ptr + row, loss)


@triton.jit
def _reduce_losses_kernel(
    row_losses_ptr,
    targets_ptr,
    output_ptr,
    n_rows,
    REDUCTION_MODE: tl.constexpr,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_SIZE)
    losses = tl.load(
        row_losses_ptr + row_offsets,
        mask=row_offsets < n_rows,
        other=0.0,
    )
    targets = tl.load(
        targets_ptr + row_offsets,
        mask=row_offsets < n_rows,
        other=ignore_index,
    )
    valid = (targets != ignore_index) & (targets >= 0)
    losses = tl.where(valid, losses, 0.0)
    total = tl.sum(losses, axis=0)
    valid_count = tl.sum(valid.to(tl.float32), axis=0)

    if REDUCTION_MODE == 0:
        result = total / valid_count
    elif REDUCTION_MODE == 1:
        result = total
    else:
        result = 0.0
    tl.store(output_ptr, result)


def _reduction_mode(reduction: str) -> int:
    if reduction == "mean":
        return 0
    if reduction == "sum":
        return 1
    raise ValueError("reduction must be 'mean' or 'sum'")


def build(context):
    def wrapper(
        logits,
        targets,
        normalized_shape,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        eps=1e-5,
        *,
        out=None,
    ):
        if logits.ndim < 2:
            raise ValueError("logits must have at least two dimensions")
        n_rows = logits.shape[0]
        n_cols = logits.shape[1]
        expected_normalized_shape = tuple(logits.shape[1:])
        if isinstance(normalized_shape, int):
            actual_normalized_shape = (normalized_shape,)
        else:
            actual_normalized_shape = tuple(normalized_shape)
        if actual_normalized_shape != expected_normalized_shape:
            raise ValueError("normalized_shape must match the class dimensions of logits")
        if logits.shape != targets.shape and targets.ndim != 0:
            raise ValueError("class-index targets must have shape (N, *)")
        if weight is not None or label_smoothing != 0.0:
            raise NotImplementedError("unsupported optional loss parameters")
        if logits.dtype != torch.float32 or targets.dtype != torch.int64:
            raise ValueError("expected float32 logits and int64 targets")

        if out is None:
            probabilities = torch.empty_like(logits)
        else:
            probabilities = out
            if probabilities.shape != logits.shape or probabilities.dtype != logits.dtype:
                raise ValueError("out must match logits in shape and dtype")

        row_losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        loss = torch.empty((), dtype=torch.float32, device=logits.device)
        block_size = triton.next_power_of_2(n_cols)
        reduction_mode = _reduction_mode(reduction)

        _fused_kernel[(n_rows,)](
            logits,
            targets,
            probabilities,
            row_losses,
            n_cols,
            logits.stride(0),
            probabilities.stride(0),
            eps,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )

        reduction_block = triton.next_power_of_2(n_rows)
        _reduce_losses_kernel[(1,)](
            row_losses,
            targets,
            loss,
            n_rows,
            ignore_index,
            REDUCTION_MODE=reduction_mode,
            BLOCK_SIZE=reduction_block,
            num_warps=4,
        )
        return loss, probabilities

    return wrapper
