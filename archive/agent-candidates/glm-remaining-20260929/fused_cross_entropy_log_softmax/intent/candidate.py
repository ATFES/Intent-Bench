import torch

import intent
import intent.language as I


@intent.kernel
def fused_row_losses(
    logits: I.In[I.f32, ("N", "C")],
    target: I.In[I.i64, ("N",)],
    class_weight: I.In[I.f32, ("C",)],
    partial: I.Out[I.f32, ("N", 2)],
    ignore_index: I.i64,
    label_smoothing: I.f32,
    USE_WEIGHT: I.Constexpr[bool],
    USE_SMOOTHING: I.Constexpr[bool],
):
    rows = I.domain(0, logits.shape[0])
    for row in I.parallel(rows):
        target_value = target[row]
        target_index = I.cast(target_value, I.index)
        ignore_index_index = I.cast(ignore_index, I.index)
        valid = target_index != ignore_index_index
        values = logits[row, :]
        maximum = I.reduce.max(values, axis=0, acc_dtype=I.f32)
        shifted = values - maximum
        logsumexp = I.log(I.reduce.sum(I.exp(shifted), axis=0, acc_dtype=I.f32))

        if USE_SMOOTHING:
            class_ids = I.indices(I.domain(0, values.shape[0]))
            target_indicator = I.cast(
                I.select(class_ids == target_index, 1.0, 0.0), I.f32
            )
            smoothing_over_classes = label_smoothing / I.cast(values.shape[0], I.f32)
            probabilities = I.full(values.shape, smoothing_over_classes, I.f32)
            probabilities = probabilities + (1.0 - label_smoothing) * target_indicator
            weighted_log_probability = I.reduce.sum(
                probabilities * shifted, axis=0, acc_dtype=I.f32
            )
            if valid:
                row_loss = logsumexp - weighted_log_probability
                if USE_WEIGHT:
                    row_loss = row_loss * class_weight[target_index]
                    row_weight = class_weight[target_index]
                else:
                    row_weight = I.cast(1.0, I.f32)
            else:
                row_loss = I.cast(0.0, I.f32)
                row_weight = I.cast(0.0, I.f32)
        else:
            if valid:
                row_loss = logsumexp - logits[row, target_index]
                if USE_WEIGHT:
                    row_weight = class_weight[target_index]
                else:
                    row_weight = I.cast(1.0, I.f32)
            else:
                row_loss = I.cast(0.0, I.f32)
                row_weight = I.cast(0.0, I.f32)

        partial[row, 0] = row_loss
        partial[row, 1] = row_weight


@intent.kernel
def reduce_partial_losses(
    partial: I.In[I.f32, ("N", 2)],
    output: I.Out[I.f32, ()],
    USE_MEAN: I.Constexpr[bool],
):
    total_loss = I.reduce.sum(partial[:, 0], axis=0, acc_dtype=I.f32)
    if USE_MEAN:
        total_weight = I.reduce.sum(partial[:, 1], axis=0, acc_dtype=I.f32)
        output[()] = total_loss / total_weight
    else:
        output[()] = total_loss


def build(context):
    row_artifacts = {
        (False, False): context.compile(
            "fused_row_losses_plain",
            fused_row_losses,
            constexprs={"USE_WEIGHT": False, "USE_SMOOTHING": False},
        ),
        (False, True): context.compile(
            "fused_row_losses_smoothed",
            fused_row_losses,
            constexprs={"USE_WEIGHT": False, "USE_SMOOTHING": True},
        ),
        (True, False): context.compile(
            "fused_row_losses_weighted",
            fused_row_losses,
            constexprs={"USE_WEIGHT": True, "USE_SMOOTHING": False},
        ),
        (True, True): context.compile(
            "fused_row_losses_weighted_smoothed",
            fused_row_losses,
            constexprs={"USE_WEIGHT": True, "USE_SMOOTHING": True},
        ),
    }
    reduction_artifacts = {
        False: context.compile(
            "reduce_partial_losses_sum",
            reduce_partial_losses,
            constexprs={"USE_MEAN": False},
        ),
        True: context.compile(
            "reduce_partial_losses_mean",
            reduce_partial_losses,
            constexprs={"USE_MEAN": True},
        ),
    }

    def fused_cross_entropy_log_softmax(
        input,
        target,
        dim=1,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
    ):
        if input.ndim != 2:
            raise ValueError("input must be a 2-D tensor")
        if dim not in (0, 1):
            raise ValueError("dim must be 0 or 1")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        logits = input if dim == 1 else input.t()
        if logits.shape[0] != target.shape[0]:
            raise ValueError("target size must match the non-softmax input dimension")
        if weight is not None and weight.shape[0] != logits.shape[1]:
            raise ValueError("weight size must match the number of classes")

        row_count = logits.shape[0]
        partial = input.new_empty((row_count, 2))
        row_artifact = row_artifacts[(weight is not None, label_smoothing != 0.0)]
        row_artifact(
            logits,
            target,
            weight if weight is not None else input.new_empty((logits.shape[1],)),
            partial,
            ignore_index,
            float(label_smoothing),
        )

        if reduction == "none":
            return partial[:, 0]

        output = input.new_empty(())
        reduction_artifact = reduction_artifacts[reduction == "mean"]
        reduction_artifact(partial, output)
        return output

    return fused_cross_entropy_log_softmax
