import torch
import intent
import intent.language as I


@intent.kernel
def _row_losses(
    input: I.In[I.f32, ("M", "C")],
    target: I.In[I.i64, ("M",)],
    output: I.Out[I.f32, ("M",)],
    valid: I.Out[I.f32, ("M",)],
    ignore_index: I.i64,
):
    M, C = input.shape
    rows = I.domain(0, M)
    classes = I.domain(0, C)

    for row in I.parallel(rows):
        logits = input[row, classes]
        maximum = I.reduce.max(logits, axis=0, acc_dtype=I.f32)
        shifted = logits - maximum
        normalizer = I.reduce.sum(I.exp(shifted), axis=0, acc_dtype=I.f32)
        logsumexp = maximum + I.log(normalizer)

        target_value = target[row]
        ignored = target_value == ignore_index
        safe_target = I.select(ignored, I.cast(0, I.i64), target_value)
        class_index = I.cast(safe_target, I.index)
        selected = input[row, class_index]
        loss = -(selected - logsumexp)
        output[row] = I.select(ignored, I.cast(0.0, I.f32), loss)
        valid[row] = I.select(ignored, I.cast(0.0, I.f32), I.cast(1.0, I.f32))


@intent.kernel
def _mean_loss(
    values: I.In[I.f32, ("M",)],
    valid: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ()],
):
    M, = values.shape
    rows = I.domain(0, M)
    total = I.reduce.sum(values[rows], axis=0, acc_dtype=I.f32)
    count = I.reduce.sum(valid[rows], axis=0, acc_dtype=I.f32)
    output[()] = total / count


def build(context):
    row_losses = context.compile("fused_ce_log_softmax_rows", _row_losses, constexprs={})
    mean_loss = context.compile("fused_ce_log_softmax_mean", _mean_loss, constexprs={})

    def fused_cross_entropy_log_softmax(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        values = torch.empty((input.shape[0],), dtype=torch.float32, device=input.device)
        valid = torch.empty((input.shape[0],), dtype=torch.float32, device=input.device)
        output = torch.empty((), dtype=torch.float32, device=input.device)
        row_losses(input, target, values, valid, ignore_index)
        mean_loss(values, valid, output)
        return output

    return fused_cross_entropy_log_softmax
