from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _row_cosine_kernel(
    input1: I.In[I.f32, ("M", "N")],
    input2: I.In[I.f32, ("M", "N")],
    cosine: I.Out[I.f32, ("M",)],
) -> None:
    epsilon = I.cast(1e-12, I.f32)
    input1_squared = I.reduce.sum(
        input1 * input1,
        axis=1,
        acc_dtype=I.f32,
    )
    input2_squared = I.reduce.sum(
        input2 * input2,
        axis=1,
        acc_dtype=I.f32,
    )
    input1_norm = I.maximum(I.sqrt(input1_squared), epsilon)
    input2_norm = I.maximum(I.sqrt(input2_squared), epsilon)
    input1_scale = I.cast(1.0, I.f32) / input1_norm
    input2_scale = I.cast(1.0, I.f32) / input2_norm
    normalized1 = input1 * I.reshape(
        input1_scale,
        (input1_scale.shape[0], 1),
    )
    normalized2 = input2 * I.reshape(
        input2_scale,
        (input2_scale.shape[0], 1),
    )
    cosine_values = I.reduce.sum(
        normalized1 * normalized2,
        axis=1,
        acc_dtype=I.f32,
    )
    rows = I.domain(0, input1.shape[0])
    cosine[rows] = cosine_values


@intent.kernel
def _none_loss_kernel(
    cosine: I.In[I.f32, ("M",)],
    target: I.In[I.f32, ("M",)],
    margin: I.f32,
    loss: I.Out[I.f32, ("M",)],
) -> None:
    rows = I.domain(0, cosine.shape[0])
    one = I.cast(1.0, I.f32)
    zero = I.cast(0.0, I.f32)
    cosine_value = cosine[rows]
    target_value = target[rows]
    row_loss = I.maximum(one - cosine_value * target_value, zero)
    if margin > I.cast(0.0, I.f32):
        row_loss = I.maximum(row_loss, margin - cosine_value)
    loss[rows] = row_loss


@intent.kernel
def _sum_loss_kernel(
    cosine: I.In[I.f32, ("M",)],
    target: I.In[I.f32, ("M",)],
    margin: I.f32,
    result: I.Out[I.f32, ()],
) -> None:
    rows = I.domain(0, cosine.shape[0])
    one = I.cast(1.0, I.f32)
    zero = I.cast(0.0, I.f32)
    cosine_value = cosine[rows]
    target_value = target[rows]
    row_loss = I.maximum(one - cosine_value * target_value, zero)
    if margin > I.cast(0.0, I.f32):
        row_loss = I.maximum(row_loss, margin - cosine_value)
    total = I.reduce.sum(row_loss, axis=0, acc_dtype=I.f32)
    result[()] = total


@intent.kernel
def _mean_loss_kernel(
    cosine: I.In[I.f32, ("M",)],
    target: I.In[I.f32, ("M",)],
    margin: I.f32,
    row_count: I.f32,
    result: I.Out[I.f32, ()],
) -> None:
    rows = I.domain(0, cosine.shape[0])
    one = I.cast(1.0, I.f32)
    zero = I.cast(0.0, I.f32)
    cosine_value = cosine[rows]
    target_value = target[rows]
    row_loss = I.maximum(one - cosine_value * target_value, zero)
    if margin > I.cast(0.0, I.f32):
        row_loss = I.maximum(row_loss, margin - cosine_value)
    total = I.reduce.sum(row_loss, axis=0, acc_dtype=I.f32)
    result[()] = total / row_count


def build(context):
    row_cosine = context.compile("fused_cosine_rows", _row_cosine_kernel)
    none_loss = context.compile("fused_cosine_loss_none", _none_loss_kernel)
    sum_loss = context.compile("fused_cosine_loss_sum", _sum_loss_kernel)
    mean_loss = context.compile("fused_cosine_loss_mean", _mean_loss_kernel)

    def fused_cosine_embedding_loss_with_normalization(
        input1,
        input2,
        target,
        margin: float = 0,
        reduction: str = "mean",
    ):
        if input1.shape != input2.shape or input1.ndim != 2:
            raise ValueError("input1 and input2 must be matching rank-2 tensors")
        if target.ndim != 1 or target.shape[0] != input1.shape[0]:
            raise ValueError("target must contain one value per input row")
        if input1.dtype != input2.dtype or input1.dtype != target.dtype:
            raise ValueError("input1, input2, and target must have the same dtype")

        row_count = input1.shape[0]
        cosine = input1.new_empty((row_count,), dtype=input1.dtype)
        row_cosine(input1, input2, cosine)

        if reduction == "none":
            output = input1.new_empty((row_count,), dtype=input1.dtype)
            none_loss(cosine, target, margin, output)
            return output
        if reduction == "sum":
            output = input1.new_empty((), dtype=input1.dtype)
            sum_loss(cosine, target, margin, output)
            return output
        if reduction == "mean":
            output = input1.new_empty((), dtype=input1.dtype)
            mean_loss(cosine, target, margin, float(row_count), output)
            return output
        raise ValueError(f"unsupported reduction: {reduction!r}")

    return fused_cosine_embedding_loss_with_normalization
