import torch
import intent
import intent.language as I


@intent.kernel
def row_losses(
    input: I.In[I.f32, (4096, 128)],
    target: I.In[I.i64, (4096,)],
    losses: I.Out[I.f32, (4096,)],
    counts: I.Out[I.f32, (4096,)],
    ignore_index: I.i64,
    label_smoothing: I.f32,
):
    for row in I.parallel(I.domain(0, 4096)):
        values = input[row, :]
        shifted = values - I.reduce.max(values, axis=0)
        log_probs = shifted - I.log(I.reduce.sum(I.exp(shifted), axis=0))
        label = target[row]
        valid = label != ignore_index
        log_prob = I.gather(log_probs, I.cast(label, I.index), valid=valid,
                            fill=0.0)
        smooth = -I.reduce.sum(log_probs, axis=0) / 128.0
        loss = -(1.0 - label_smoothing) * log_prob + label_smoothing * smooth
        losses[row] = I.select(valid, loss, 0.0)
        counts[row] = I.select(valid, I.cast(1.0, I.f32), I.cast(0.0, I.f32))


@intent.kernel
def finish_loss(
    losses: I.In[I.f32, (4096,)],
    counts: I.In[I.f32, (4096,)],
    out: I.Out[I.f32, ()],
    MEAN: I.Constexpr[bool],
):
    total = I.reduce.sum(losses[:], axis=0)
    if MEAN:
        total = total / I.reduce.sum(counts[:], axis=0)
    out[()] = total


def build(context):
    rows = context.compile("row_losses", row_losses)
    finals = {mean: context.compile("finish_loss_" + str(mean), finish_loss,
                                    constexprs={"MEAN": mean})
              for mean in (False, True)}

    def wrapper(input, target, dim=1, weight=None, ignore_index=-100,
                reduction="mean", label_smoothing=0.0):
        if weight is not None or dim not in (1, -1):
            raise NotImplementedError("this program supports unweighted class-axis loss")
        losses = torch.empty((4096,), dtype=input.dtype, device=input.device)
        counts = torch.empty_like(losses)
        rows(input, target, losses, counts, ignore_index, float(label_smoothing))
        if reduction == "none":
            return losses
        if reduction not in ("sum", "mean"):
            raise ValueError("invalid reduction")
        out = torch.empty((), dtype=input.dtype, device=input.device)
        finals[reduction == "mean"](losses, counts, out)
        return out

    return wrapper
