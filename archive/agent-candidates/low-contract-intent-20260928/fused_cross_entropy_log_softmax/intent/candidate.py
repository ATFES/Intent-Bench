import torch
import intent
import intent.language as I


@intent.kernel
def row_loss(
    x: I.In[I.f32, ("M", "C")],
    target: I.In[I.i64, ("M",)],
    partial: I.Out[I.f32, ("M",)],
    weights: I.Out[I.f32, ("M",)],
    ignore_index: I.i64,
):
    rows = I.domain(0, x.shape[0])
    for r in I.parallel(rows):
        values = x[r, :]
        maximum = I.reduce.max(values, axis=0, acc_dtype=I.f32)
        shifted = values - maximum
        denom = I.reduce.sum(I.exp(shifted), axis=0, acc_dtype=I.f32)
        log_denom = I.log(denom)
        cls = I.cast(target[r], I.index)
        ignored = target[r] == ignore_index
        selected = I.gather(values, cls, valid=ignored == False, fill=I.cast(0.0, I.f32))
        loss = (maximum + log_denom) - selected
        partial[r] = I.select(ignored, I.cast(0.0, I.f32), loss)
        weights[r] = I.select(ignored, I.cast(0.0, I.f32), I.cast(1.0, I.f32))


@intent.kernel
def reduce_loss(
    partial: I.In[I.f32, ("M",)],
    weights: I.In[I.f32, ("M",)],
    out: I.Out[I.f32, ()],
    denominator: I.Out[I.f32, ()],
):
    values = partial[:]
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    count = I.reduce.sum(weights[:], axis=0, acc_dtype=I.f32)
    out[()] = total / count
    denominator[()] = count


def build(context):
    row_artifact = context.compile("fused_ce_rows", row_loss)
    reduce_artifact = context.compile("fused_ce_reduce", reduce_loss)

    def fused_cross_entropy_log_softmax(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        partial = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
        weights = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
        output = torch.empty((), device=input.device, dtype=torch.float32)
        denominator = torch.empty((), device=input.device, dtype=torch.float32)
        row_artifact(input, target, partial, weights, ignore_index)
        reduce_artifact(partial, weights, output, denominator)
        return output

    return fused_cross_entropy_log_softmax
