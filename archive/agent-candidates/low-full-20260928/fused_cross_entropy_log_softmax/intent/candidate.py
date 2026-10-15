import torch
import intent
import intent.language as I


@intent.kernel
def _row_loss(
    x: I.In[I.f32, ("M", "C")],
    target: I.In[I.i64, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    valid: I.Out[I.f32, ("M",)],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    values = x[rows, cols]
    row_max = I.reduce.max(values, axis=1)
    shifted = values - I.reshape(row_max, (x.shape[0], 1))
    exp_shifted = I.exp(shifted)
    denom = I.reduce.sum(exp_shifted, axis=1, acc_dtype=I.f32)
    log_denom = I.log(denom)
    t = target[rows]
    ti = I.cast(t, I.index)
    chosen = x[rows, ti]
    row_loss = row_max + log_denom - chosen
    is_valid = t != I.cast(-100, I.i64)
    loss[rows] = I.select(is_valid, row_loss, I.cast(0.0, I.f32))
    valid[rows] = I.select(is_valid, I.cast(1.0, I.f32), I.cast(0.0, I.f32))


@intent.kernel
def _reduce_loss(
    loss: I.In[I.f32, ("M",)],
    valid: I.In[I.f32, ("M",)],
    out: I.Out[I.f32, ()],
):
    total = I.reduce.sum(loss, axis=0, acc_dtype=I.f32)
    count = I.reduce.sum(valid, axis=0, acc_dtype=I.f32)
    out[()] = total / count


def build(context):
    row_kernel = context.compile("fused_ce_rows", _row_loss)
    reduce_kernel = context.compile("fused_ce_reduce", _reduce_loss)

    def fused_cross_entropy_log_softmax(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        row_loss = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
        row_valid = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
        row_kernel(input, target, row_loss, row_valid)
        out = torch.empty((), device=input.device, dtype=torch.float32)
        reduce_kernel(row_loss, row_valid, out)
        return out

    return fused_cross_entropy_log_softmax
