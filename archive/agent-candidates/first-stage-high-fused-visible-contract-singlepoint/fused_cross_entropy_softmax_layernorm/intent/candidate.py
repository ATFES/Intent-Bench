import torch
import intent
import intent.language as I


@intent.kernel
def _row_kernel(
    logits: I.In[I.f32, ("M", "C")],
    targets: I.In[I.i64, ("M",)],
    output: I.Out[I.f32, ("M", "C")],
    row_losses: I.Out[I.f32, ("M",)],
    EPS: I.Constexpr[float],
    IGNORE_INDEX: I.Constexpr[int],
):
    M, C = logits.shape
    rows = I.domain(0, M)
    cols = I.domain(0, C)
    zero = I.cast(0.0, I.f32)
    class_count = I.cast(C, I.f32)

    for row in I.parallel(rows):
        row_logits = logits[row, cols]
        row_max = I.reduce.max(row_logits, axis=0, acc_dtype=I.f32)
        shifted = row_logits - row_max
        exp_shifted = I.exp(shifted)
        exp_sum = I.reduce.sum(exp_shifted, axis=0, acc_dtype=I.f32)
        probabilities = I.fdiv(exp_shifted, exp_sum)

        probability_sum = I.reduce.sum(probabilities, axis=0, acc_dtype=I.f32)
        mean = I.fdiv(probability_sum, class_count)
        centered = probabilities - mean
        variance_sum = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32)
        variance = I.fdiv(variance_sum, class_count)
        normalized = I.fdiv(centered, I.sqrt(variance + I.cast(EPS, I.f32)))
        output[row, cols] = normalized

        target = targets[row]
        if target == IGNORE_INDEX:
            row_losses[row] = zero
        else:
            target_index = I.cast(target, I.index)
            target_logit = logits[row, target_index]
            log_normalizer = I.log(exp_sum) + row_max
            row_losses[row] = log_normalizer - target_logit


@intent.kernel
def _mean_loss_kernel(
    row_losses: I.In[I.f32, ("M",)],
    targets: I.In[I.i64, ("M",)],
    loss: I.Out[I.f32, ()],
    IGNORE_INDEX: I.Constexpr[int],
):
    M = row_losses.shape[0]
    rows = I.domain(0, M)
    zero = I.cast(0.0, I.f32)
    valid = targets[rows] != IGNORE_INDEX
    masked_losses = I.select(valid, row_losses[rows], zero)
    total = I.reduce.sum(masked_losses, axis=0, acc_dtype=I.f32)
    count = I.reduce.sum(I.cast(valid, I.f32), axis=0, acc_dtype=I.f32)
    loss[()] = I.fdiv(total, count)


def build(context):
    row_artifact = context.compile(
        "fused_cross_entropy_softmax_layernorm_rows",
        _row_kernel,
        constexprs={"EPS": 1.0e-5, "IGNORE_INDEX": -100},
    )
    loss_artifact = context.compile(
        "fused_cross_entropy_softmax_layernorm_mean",
        _mean_loss_kernel,
        constexprs={"IGNORE_INDEX": -100},
    )

    def fused_cross_entropy_softmax_layernorm(
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
        if out is None:
            output = torch.empty_like(logits)
        else:
            output = out
        row_losses = torch.empty(
            (logits.shape[0],), dtype=torch.float32, device=logits.device
        )
        loss = torch.empty((), dtype=torch.float32, device=logits.device)
        row_artifact(logits, targets, output, row_losses)
        loss_artifact(row_losses, targets, loss)
        return loss, output

    return fused_cross_entropy_softmax_layernorm
