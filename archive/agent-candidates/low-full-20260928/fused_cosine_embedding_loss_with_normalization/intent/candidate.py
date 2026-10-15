import torch
import intent
import intent.language as I


@intent.kernel
def _row_loss(
    input1: I.In[I.f32, ("M", "K")],
    input2: I.In[I.f32, ("M", "K")],
    target: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ("M",)],
    margin: I.f32,
):
    rows = I.domain(0, input1.shape[0])
    cols = I.domain(0, input1.shape[1])
    a = input1[rows, cols]
    b = input2[rows, cols]
    aa = I.reduce.sum(a * a, axis=1, acc_dtype=I.f32)
    bb = I.reduce.sum(b * b, axis=1, acc_dtype=I.f32)
    eps = I.cast(1.0e-12, I.f32)
    na = I.sqrt(I.select(aa > eps, aa, eps))
    nb = I.sqrt(I.select(bb > eps, bb, eps))
    ua = a / I.reshape(na, (input1.shape[0], 1))
    ub = b / I.reshape(nb, (input1.shape[0], 1))
    cosine = I.reduce.sum(ua * ub, axis=1, acc_dtype=I.f32)
    base = I.select(I.cast(1.0, I.f32) - cosine * target > I.cast(0.0, I.f32),
                    I.cast(1.0, I.f32) - cosine * target, I.cast(0.0, I.f32))
    margin_loss = margin - cosine
    loss = I.select(margin > I.cast(0.0, I.f32),
                    I.select(base > margin_loss, base, margin_loss), base)
    output[rows] = loss


@intent.kernel
def _reduce_loss(
    values: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ()],
    mean: I.Constexpr[bool],
):
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    if mean:
        total = total / I.cast(values.shape[0], I.f32)
    output[()] = total


def build(context):
    row_artifact = context.compile("cosine_row_loss", _row_loss)
    sum_artifact = context.compile("cosine_loss_sum", _reduce_loss, constexprs={"mean": False})
    mean_artifact = context.compile("cosine_loss_mean", _reduce_loss, constexprs={"mean": True})

    def fused_cosine_embedding_loss_with_normalization(
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        rows = torch.empty((input1.shape[0],), device=input1.device, dtype=input1.dtype)
        row_artifact(input1, input2, target, rows, margin)
        if reduction == "none":
            return rows
        result = torch.empty((), device=input1.device, dtype=input1.dtype)
        if reduction == "sum":
            sum_artifact(rows, result)
        else:
            mean_artifact(rows, result)
        return result

    return fused_cosine_embedding_loss_with_normalization
