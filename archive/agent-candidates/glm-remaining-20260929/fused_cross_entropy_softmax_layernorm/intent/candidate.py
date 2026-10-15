from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def fused_rows(
    logits: I.In[I.f32, ("N", "C")],
    targets: I.In[I.i64, ("N",)],
    normalized: I.Out[I.f32, ("N", "C")],
    row_losses: I.Out[I.f32, ("N",)],
    row_valid: I.Out[I.f32, ("N",)],
    ignore_index: I.i64,
    label_smoothing: I.f32,
    eps: I.f32,
) -> None:
    N, C = logits.shape
    rows = I.domain(0, N)
    classes = I.domain(0, C)
    row_values = logits[rows, classes]
    class_indices = I.indices(classes)

    maxima = I.reduce.max(row_values, axis=1, acc_dtype=I.f32)
    shifted = I.exp(row_values - I.reshape(maxima, (N, 1)))
    sums = I.reduce.sum(shifted, axis=1, acc_dtype=I.f32)
    probabilities = shifted / I.reshape(sums, (N, 1))

    class_count = I.cast(C, I.f32)
    mean_probability = I.cast(1.0, I.f32) / class_count
    probability_squares = I.reduce.sum(
        probabilities * probabilities, axis=1, acc_dtype=I.f32
    )
    variance = probability_squares / class_count - mean_probability * mean_probability
    normalization_scale = I.rsqrt(variance + eps)
    normalized[rows, classes] = (
        probabilities - mean_probability
    ) * I.reshape(normalization_scale, (N, 1))

    target_mask = I.cast(I.reshape(class_indices, (1, C)), I.i64) == I.reshape(
        targets, (N, 1)
    )
    target_logits = I.reduce.sum(
        I.mask(row_values, target_mask, I.cast(0.0, I.f32)),
        axis=1,
        acc_dtype=I.f32,
    )
    log_normalizers = I.log(sums) + maxima
    if label_smoothing > I.cast(0.0, I.f32):
        row_sums = I.reduce.sum(row_values, axis=1, acc_dtype=I.f32)
        row_losses_unfiltered = (
            (I.cast(1.0, I.f32) - label_smoothing)
            * (log_normalizers - target_logits)
            + label_smoothing
            * (class_count * log_normalizers - row_sums)
            / class_count
        )
    else:
        row_losses_unfiltered = log_normalizers - target_logits

    valid = targets != ignore_index
    row_losses[rows] = I.select(
        valid, row_losses_unfiltered, I.cast(0.0, I.f32)
    )
    row_valid[rows] = I.select(valid, I.cast(1.0, I.f32), I.cast(0.0, I.f32))


@intent.kernel
def reduce_losses(
    row_losses: I.In[I.f32, ("N",)],
    row_valid: I.In[I.f32, ("N",)],
    loss: I.Out[I.f32, ()],
    reduction: I.i64,
) -> None:
    N = row_losses.shape[0]
    rows = I.domain(0, N)
    losses = row_losses[rows]
    valid_values = row_valid[rows]
    loss_total = I.reduce.sum(losses, axis=0, acc_dtype=I.f32)
    valid_total = I.reduce.sum(valid_values, axis=0, acc_dtype=I.f32)
    mean_count = I.select(
        valid_total > I.cast(0.0, I.f32),
        valid_total,
        I.cast(1.0, I.f32),
    )
    reduction_value = I.select(
        reduction == I.cast(0, I.i64),
        loss_total,
        I.select(
            reduction == I.cast(1, I.i64),
            loss_total / mean_count,
            loss_total,
        ),
    )
    loss[()] = reduction_value


def build(context):
    normalized = torch.empty((256, 1024), dtype=torch.float32, device="cuda")
    row_losses = torch.empty((256,), dtype=torch.float32, device="cuda")
    row_valid = torch.empty((256,), dtype=torch.float32, device="cuda")
    loss = torch.empty((), dtype=torch.float32, device="cuda")
    fused = context.compile("fused_cross_entropy_softmax_layernorm_rows", fused_rows)
    reducer = context.compile("fused_cross_entropy_softmax_layernorm_reduce", reduce_losses)

    reduction_modes = {"none": 0, "mean": 1, "sum": 2}

    def fused_cross_entropy_softmax_layernorm(
        logits,
        targets,
        normalized_shape,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        eps=1e-05,
        *,
        out=None,
    ):
        if logits.ndim != 2 or logits.dtype != torch.float32:
            raise ValueError("this profile requires float32 logits of shape (N, C)")
        if targets.dtype != torch.int64 or targets.shape != logits.shape[:1]:
            raise ValueError("targets must be int64 indices with shape (N,)")
        normalization_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else torch.Size(normalized_shape)
        )
        if tuple(normalization_shape) != (logits.shape[1],):
            raise ValueError("normalized_shape must equal the class dimension")
        if weight is not None:
            raise ValueError("this fixed profile does not accept class weights")
        if reduction not in reduction_modes:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        destination = normalized if out is None else out
        fused(
            logits,
            targets,
            destination,
            row_losses,
            row_valid,
            ignore_index,
            label_smoothing,
            eps,
        )
        reducer(row_losses, row_valid, loss, reduction_modes[reduction])
        return loss, destination

    return fused_cross_entropy_softmax_layernorm
