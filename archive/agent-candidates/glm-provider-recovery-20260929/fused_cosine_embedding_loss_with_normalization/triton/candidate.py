import torch
import triton
import triton.language as tl


@triton.jit
def _row_loss_kernel(
    input1_ptr,
    input2_ptr,
    target_ptr,
    loss_ptr,
    row_stride,
    margin,
    BLOCK_SIZE: tl.constexpr,
    HAS_MARGIN: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    offsets = row * row_stride + columns
    valid = columns < BLOCK_SIZE

    x = tl.load(input1_ptr + offsets, mask=valid, other=0.0)
    y = tl.load(input2_ptr + offsets, mask=valid, other=0.0)
    x = x.to(tl.float32)
    y = y.to(tl.float32)

    norm_x = tl.maximum(tl.sqrt(tl.sum(x * x)), 1e-12)
    norm_y = tl.maximum(tl.sqrt(tl.sum(y * y)), 1e-12)
    u = x / norm_x
    v = y / norm_y
    cosine = tl.sum(u * v)
    target = tl.load(target_ptr + row).to(tl.float32)
    loss = tl.maximum(1.0 - cosine * target, 0.0)
    if HAS_MARGIN:
        loss = tl.maximum(loss, margin - cosine)

    tl.store(loss_ptr + row, loss)


@triton.jit
def _reduce_loss_kernel(
    loss_ptr,
    output_ptr,
    row_count,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.program_id(0) == 0:
        total = 0.0
        for row_start in range(0, row_count, BLOCK_SIZE):
            rows = row_start + tl.arange(0, BLOCK_SIZE)
            values = tl.load(
                loss_ptr + rows,
                mask=rows < row_count,
                other=0.0,
            ).to(tl.float32)
            total += tl.sum(values)
        tl.store(output_ptr, total * scale)


def build(context):
    def fused_cosine_embedding_loss_with_normalization(
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        row_count, column_count = input1.shape
        losses = torch.empty(
            (row_count,),
            device=input1.device,
            dtype=input1.dtype,
        )

        has_margin = margin > 0
        _row_loss_kernel[(row_count,)](
            input1,
            input2,
            target,
            losses,
            input1.stride(0),
            float(margin),
            BLOCK_SIZE=triton.next_power_of_2(column_count),
            HAS_MARGIN=has_margin,
        )

        if reduction == "none":
            return losses

        result = torch.empty((), device=input1.device, dtype=input1.dtype)
        if reduction == "sum":
            scale = 1.0
        elif reduction == "mean":
            scale = 1.0 / row_count
        else:
            raise ValueError(f"invalid reduction: {reduction!r}")

        _reduce_loss_kernel[(1,)](
            losses,
            result,
            row_count,
            scale,
            BLOCK_SIZE=1024,
        )
        return result

    return fused_cosine_embedding_loss_with_normalization
