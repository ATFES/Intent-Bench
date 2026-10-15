import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_rows(
    scratch_ptr,
    output_ptr,
    n_rows,
    block_n: tl.constexpr,
    reduction: tl.constexpr,
):
    total_loss = 0.0
    total_weight = 0.0
    for start in range(0, n_rows, block_n):
        rows = start + tl.arange(0, block_n)
        valid = rows < n_rows
        loss_offsets = rows * 2 if reduction == 1 else rows
        loss = tl.load(scratch_ptr + loss_offsets, mask=valid, other=0.0)
        total_loss += tl.sum(loss, axis=0)
        if reduction == 1:
            sample_weight = tl.load(scratch_ptr + rows * 2 + 1, mask=valid, other=0.0)
            total_weight += tl.sum(sample_weight, axis=0)

    if reduction == 1:
        result = total_loss / total_weight
    else:
        result = total_loss
    tl.store(output_ptr, result)


@triton.jit
def _cross_entropy_rows(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    none_output_ptr,
    scratch_ptr,
    n_rows,
    n_classes,
    block_c: tl.constexpr,
    input_row_stride,
    ignore_index,
    label_smoothing,
    reduction: tl.constexpr,
    has_weight: tl.constexpr,
    smooth: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    classes = tl.arange(0, block_c)
    target = tl.load(target_ptr + row)
    ignored = target == ignore_index
    safe_target = tl.where(ignored, 0, target)
    logits = tl.load(input_ptr + row * input_row_stride + classes)
    row_max = tl.max(logits, axis=0)
    stable_logits = logits - row_max
    log_sum_exp = tl.log(tl.sum(tl.exp(stable_logits), axis=0)) + row_max

    if has_weight:
        class_weights = tl.load(weight_ptr + classes)
        target_weight = tl.load(weight_ptr + safe_target)
    else:
        class_weights = 1.0
        target_weight = 1.0

    if smooth:
        selected = classes == safe_target
        smoothed_weights = label_smoothing * class_weights / n_classes
        smoothed_weights = tl.where(
            selected,
            smoothed_weights + (1.0 - label_smoothing) * target_weight,
            smoothed_weights,
        )
        row_loss = tl.sum(smoothed_weights * (log_sum_exp - logits), axis=0)
        row_weight = target_weight
    else:
        row_loss = target_weight * (log_sum_exp - tl.sum(
            tl.where(classes == safe_target, logits, 0.0), axis=0
        ))
        row_weight = target_weight

    row_loss = tl.where(ignored, 0.0, row_loss)
    row_weight = tl.where(ignored, 0.0, row_weight)

    if reduction == 0:
        tl.store(none_output_ptr + row, row_loss)
    elif reduction == 1:
        tl.store(scratch_ptr + row * 2, row_loss)
        tl.store(scratch_ptr + row * 2 + 1, row_weight)
    else:
        tl.store(scratch_ptr + row, row_loss)


def build(context):
    def fused_cross_entropy_log_softmax(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"invalid reduction: {reduction!r}")

        logits = input
        class_dim = dim
        if class_dim < 0:
            class_dim += logits.ndim
        n_classes = logits.shape[class_dim]
        n_rows = logits.numel() // n_classes

        if logits.ndim != 2 or class_dim not in (0, 1):
            raise ValueError("input must be two-dimensional with a valid class dimension")
        if class_dim == 0:
            logits = logits.t()
            n_classes, n_rows = n_rows, n_classes

        row_stride = logits.stride(0)
        if row_stride < logits.shape[1] or logits.stride(1) != 1:
            raise ValueError("input must have a contiguous class dimension")
        if target.stride(0) != 1:
            raise ValueError("target must be contiguous")

        reduction_code = {"none": 0, "mean": 1, "sum": 2}[reduction]
        smoothing = float(label_smoothing)
        has_weight = weight is not None
        block_c = triton.next_power_of_2(n_classes)

        if reduction == "none":
            output = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
            scratch = output
        else:
            scratch_width = 2 if reduction == "mean" else 1
            scratch = torch.empty(
                n_rows * scratch_width, device=logits.device, dtype=torch.float32
            )
            output = torch.empty((), device=logits.device, dtype=torch.float32)

            if reduction == "mean":
                reduction_grid = (1,)
                reduction_block = min(triton.next_power_of_2(n_rows), 1024)

        _cross_entropy_rows[(n_rows,)](
            logits,
            target,
            weight if weight is not None else logits,
            output,
            output,
            scratch,
            n_rows,
            n_classes,
            block_c,
            row_stride,
            ignore_index,
            smoothing,
            reduction_code,
            has_weight=has_weight,
            smooth=smoothing != 0.0,
        )

        if reduction == "sum":
            _reduce_rows[(1,)](
                scratch,
                output,
                n_rows,
                BLOCK_N=min(triton.next_power_of_2(n_rows), 1024),
                reduction=2,
            )
        elif reduction == "mean":
            _reduce_rows[reduction_grid](
                scratch,
                output,
                n_rows,
                BLOCK_N=reduction_block,
                reduction=1,
            )
        if reduction == "none":
            output_shape = list(logits.shape)
            del output_shape[-1]
            output = output.view(output_shape)
        return output

    return fused_cross_entropy_log_softmax
