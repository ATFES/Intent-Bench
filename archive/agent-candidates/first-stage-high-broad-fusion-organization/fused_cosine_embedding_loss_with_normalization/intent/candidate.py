import torch
import intent
import intent.language as I


@intent.kernel
def _row_loss(
    input1: I.In[I.f32, ("M", "N")],
    input2: I.In[I.f32, ("M", "N")],
    target: I.In[I.f32, ("M",)],
    losses: I.Out[I.f32, ("M",)],
    margin: I.f32,
):
    M, N = input1.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    epsilon = I.cast(1.0e-12, I.f32)

    for row in I.parallel(rows):
        x1 = input1[row, columns]
        x2 = input2[row, columns]

        norm1 = I.sqrt(I.reduce.sum(x1 * x1, axis=0, acc_dtype=I.f32))
        norm2 = I.sqrt(I.reduce.sum(x2 * x2, axis=0, acc_dtype=I.f32))
        denom1 = I.maximum(norm1, epsilon)
        denom2 = I.maximum(norm2, epsilon)
        unit1 = I.fdiv(x1, denom1)
        unit2 = I.fdiv(x2, denom2)
        cosine = I.reduce.sum(unit1 * unit2, axis=0, acc_dtype=I.f32)

        loss = I.maximum(
            I.cast(1.0, I.f32) - cosine * target[row],
            I.cast(0.0, I.f32),
        )
        if margin > I.cast(0.0, I.f32):
            loss = I.maximum(loss, margin - cosine)
        losses[row] = loss


@intent.kernel
def _reduce_loss(
    losses: I.In[I.f32, ("M",)],
    result: I.Out[I.f32, ()],
    scale: I.f32,
):
    M = losses.shape[0]
    rows = I.domain(0, M)
    total = I.reduce.sum(losses[rows], axis=0, acc_dtype=I.f32)
    result[()] = total * scale


def build(context):
    row_loss = context.compile("fused_cosine_row_loss", _row_loss, constexprs={})
    reduce_loss = context.compile("fused_cosine_reduce_loss", _reduce_loss, constexprs={})

    def fused_cosine_embedding_loss_with_normalization(
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        losses = torch.empty(
            (input1.shape[0],),
            device=input1.device,
            dtype=input1.dtype,
        )
        row_loss(input1, input2, target, losses, float(margin))

        if reduction == "none":
            return losses

        result = torch.empty((), device=input1.device, dtype=input1.dtype)
        if reduction == "sum":
            scale = 1.0
        elif reduction == "mean":
            scale = 1.0 / input1.shape[0]
        else:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        reduce_loss(losses, result, scale)
        return result

    return fused_cosine_embedding_loss_with_normalization
