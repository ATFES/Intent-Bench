import torch
import triton
import triton.language as tl


@triton.jit
def _loss_kernel(x, y, target, out, n_rows, n_cols, margin,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    xrow = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    yrow = tl.load(y + row * n_cols + cols, mask=mask, other=0.0)
    xrow = xrow.to(tl.float32)
    yrow = yrow.to(tl.float32)
    nx = tl.sqrt(tl.sum(xrow * xrow, axis=0))
    ny = tl.sqrt(tl.sum(yrow * yrow, axis=0))
    nx = tl.maximum(nx, 1.0e-12)
    ny = tl.maximum(ny, 1.0e-12)
    cosine = tl.sum((xrow / nx) * (yrow / ny), axis=0)
    t = tl.load(target + row).to(tl.float32)
    loss = tl.maximum(1.0 - cosine * t, 0.0)
    margin_loss = tl.maximum(loss, margin - cosine)
    loss = tl.where(margin > 0.0, margin_loss, loss)
    tl.store(out + row, loss)


@triton.jit
def _reduce_kernel(values, out, n, mean: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    vals = tl.load(values + offsets, mask=offsets < n, other=0.0)
    total = tl.sum(vals, axis=0)
    if mean:
        total = total / n
    tl.store(out, total)


def build(context):
    def wrapper(input1, input2, target, margin=0, reduction="mean"):
        n_rows = input1.shape[0]
        n_cols = input1.shape[1]
        if reduction == "none":
            result = torch.empty((n_rows,), device=input1.device, dtype=input1.dtype)
            _loss_kernel[(n_rows,)](input1, input2, target, result, n_rows, n_cols,
                                    margin, BLOCK=256)
            return result

        values = torch.empty((n_rows,), device=input1.device, dtype=input1.dtype)
        _loss_kernel[(n_rows,)](input1, input2, target, values, n_rows, n_cols,
                                margin, BLOCK=256)
        result = torch.empty((), device=input1.device, dtype=input1.dtype)
        _reduce_kernel[(1,)](values, result, n_rows, reduction == "mean", BLOCK=4096)
        return result

    return wrapper
