import torch
import intent
import intent.language as I


@intent.kernel
def _rows(
    logits: I.In[I.f32, ("M", "C")],
    targets: I.In[I.i64, ("M",)],
    row_loss: I.Out[I.f32, ("M",)],
    normalized: I.Out[I.f32, ("M", "C")],
    eps: I.f32,
):
    rows = I.domain(0, logits.shape[0])
    cols = I.domain(0, logits.shape[1])
    for r in I.parallel(rows):
        z = logits[r, cols]
        zmax = I.reduce.max(z, axis=0)
        shifted = z - zmax
        ex = I.exp(shifted)
        denom = I.reduce.sum(ex, axis=0, acc_dtype=I.f32)
        probs = ex / denom

        mean = I.reduce.sum(probs, axis=0, acc_dtype=I.f32) / I.cast(logits.shape[1], I.f32)
        centered = probs - mean
        variance = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32) / I.cast(logits.shape[1], I.f32)
        inv_std = I.cast(1.0, I.f32) / I.sqrt(variance + eps)
        normalized[r, cols] = centered * inv_std

        target = targets[r]
        target_logit = logits[r, target]
        row_loss[r] = zmax + I.log(denom) - target_logit


@intent.kernel
def _mean_loss(
    row_loss: I.In[I.f32, ("M",)],
    result: I.Out[I.f32, ()],
):
    total = I.reduce.sum(row_loss[:], axis=0, acc_dtype=I.f32)
    result[()] = total / I.cast(row_loss.shape[0], I.f32)


def build(context):
    rows = context.compile("fused_ce_rows", _rows)
    mean_loss = context.compile("fused_ce_mean", _mean_loss)

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
        normalized = out if out is not None else torch.empty_like(logits)
        row_loss = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.float32)
        loss = torch.empty((), device=logits.device, dtype=torch.float32)
        rows(logits, targets, row_loss, normalized, eps)
        mean_loss(row_loss, loss)
        return loss, normalized

    return fused_cross_entropy_softmax_layernorm
