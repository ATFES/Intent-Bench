from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_row(
    logits_ptr,
    targets_ptr,
    output_ptr,
    row_losses_ptr,
    n_cols,
    eps,
    ignore_index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    row_logits = logits_ptr + row * n_cols + cols
    logits = tl.load(row_logits, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - row_max)
    exp_logits = tl.where(mask, exp_logits, 0.0)
    exp_sum = tl.sum(exp_logits, axis=0)
    probs = exp_logits / exp_sum

    target = tl.load(targets_ptr + row)
    target_logit = tl.sum(tl.where(cols == target, logits, 0.0), axis=0)
    row_loss = row_max + tl.log(exp_sum) - target_logit
    is_valid = target != ignore_index
    row_loss = tl.where(is_valid, row_loss, 0.0)
    tl.store(row_losses_ptr + row, row_loss)

    mean = tl.sum(probs, axis=0) / n_cols
    centered = probs - mean
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = centered * tl.rsqrt(variance + eps)
    tl.store(output_ptr + row * n_cols + cols, normalized, mask=mask)


@triton.jit
def _reduce_mean(
    row_losses_ptr,
    targets_ptr,
    loss_ptr,
    n_rows,
    ignore_index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_rows
    losses = tl.load(row_losses_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=ignore_index)
    valid = (targets != ignore_index).to(tl.float32)
    total = tl.sum(losses, axis=0)
    count = tl.sum(valid, axis=0)
    tl.store(loss_ptr, total / count)


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
        n_rows = logits.shape[0]
        n_cols = logits.shape[1]
        normalized = torch.empty_like(logits) if out is None else out
        loss = torch.empty((), dtype=torch.float32, device=logits.device)
        row_losses = torch.empty((n_rows,), dtype=torch.float32, device=logits.device)

        _fused_row[(n_rows,)](
            logits,
            targets,
            normalized,
            row_losses,
            n_cols,
            eps,
            ignore_index=ignore_index,
            BLOCK=1024,
        )
        _reduce_mean[(1,)](
            row_losses,
            targets,
            loss,
            n_rows,
            ignore_index=ignore_index,
            BLOCK=256,
        )
        return loss, normalized

    return wrapper
