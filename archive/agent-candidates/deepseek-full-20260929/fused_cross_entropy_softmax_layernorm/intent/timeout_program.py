from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _fused_row_kernel(
    logits: I.In[I.f32, ("M", "N")],
    targets: I.In[I.i64, ("M",)],
    out: I.Out[I.f32, ("M", "N")],
    row_loss: I.Out[I.f32, ("M",)],
    row_count: I.Out[I.f32, ("M",)],
    ignore_index: I.i64,
    eps: I.f32,
):
    M, N = logits.shape
    rows = I.domain(0, M)
    cols = I.domain(0, N)

    z = logits[rows, cols]

    row_max = I.reduce.max(z, axis=1)
    shifted = z - I.reshape(row_max, (M, 1))
    exp_z = I.exp(shifted)
    row_sum = I.reduce.sum(exp_z, axis=1)

    probs = exp_z / I.reshape(row_sum, (M, 1))

    prob_sum = I.reduce.sum(probs, axis=1)
    n_f = I.cast(N, I.f32)
    mean = prob_sum / n_f
    diff = probs - I.reshape(mean, (M, 1))
    var = I.reduce.sum(diff * diff, axis=1) / n_f
    inv_std = I.rsqrt(var + eps)
    normalized = diff * I.reshape(inv_std, (M, 1))
    out[rows, cols] = normalized

    logsumexp = row_max + I.log(row_sum)

    row_indices = I.indices(rows)
    target_values = targets[rows]
    target_indices = I.cast(target_values, I.index)
    valid = target_values != ignore_index
    safe_target = I.select(valid, target_indices, I.cast(0, I.index))
    selected = z[row_indices, safe_target]

    zero_f = I.cast(0.0, I.f32)
    one_f = I.cast(1.0, I.f32)
    row_loss[rows] = I.select(valid, logsumexp - selected, zero_f)
    row_count[rows] = I.select(valid, one_f, zero_f)


@intent.kernel
def _loss_reduce_kernel(
    row_loss: I.In[I.f32, ("M",)],
    row_count: I.In[I.f32, ("M",)],
    loss: I.Out[I.f32, ()],
):
    M, = row_loss.shape
    rows = I.domain(0, M)

    loss_values = row_loss[rows]
    count_values = row_count[rows]

    total_loss = I.reduce.sum(loss_values, axis=0)
    total_count = I.reduce.sum(count_values, axis=0)
    loss[()] = total_loss / total_count


def build(context):
    row_artifact = context.compile(
        "fused_cross_entropy_softmax_layernorm_row",
        _fused_row_kernel,
        constexprs={},
    )
    loss_artifact = context.compile(
        "fused_cross_entropy_softmax_layernorm_loss",
        _loss_reduce_kernel,
        constexprs={},
    )

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
        M, N = logits.shape
        if out is None:
            out = torch.empty(
                (M, N), device=logits.device, dtype=torch.float32
            )

        row_loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
        row_count = torch.empty((M,), device=logits.device, dtype=torch.float32)
        loss = torch.empty((), device=logits.device, dtype=torch.float32)

        row_artifact(
            logits,
            targets,
            out,
            row_loss,
            row_count,
            int(ignore_index),
            float(eps),
        )
        loss_artifact(row_loss, row_count, loss)

        return loss, out

    return fused_cross_entropy_softmax_layernorm
