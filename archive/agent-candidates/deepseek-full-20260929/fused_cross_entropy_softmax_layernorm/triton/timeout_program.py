import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 1024}, num_warps=4),
        triton.Config({"BLOCK_C": 1024}, num_warps=8),
        triton.Config({"BLOCK_C": 1024}, num_warps=16),
        triton.Config({"BLOCK_C": 1024}, num_warps=32),
    ],
    key=["C"],
)
@triton.jit
def fused_row_kernel(
    logits_ptr,
    targets_ptr,
    row_loss_ptr,
    out_ptr,
    weight_ptr,
    row_denom_ptr,
    N,
    C,
    stride0,
    stride1,
    out_stride0,
    out_stride1,
    target_stride,
    ignore_index,
    eps,
    HAS_WEIGHT: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < C

    base = logits_ptr + row * stride0 + cols * stride1
    x = tl.load(base, mask=col_mask, other=-float("inf"))

    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    s = tl.sum(e, axis=0)
    p = e / s

    sum_p = tl.sum(p, axis=0)
    mean = sum_p / C
    diff = p - mean
    var = tl.sum(diff * diff, axis=0) / C
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    out_base = out_ptr + row * out_stride0 + cols * out_stride1
    tl.store(out_base, (p - mean) * rstd, mask=col_mask)

    target = tl.load(targets_ptr + row * target_stride)
    ignore = ignore_index.to(tl.int64)
    valid = target != ignore
    safe_target = tl.where(valid, target, tl.zeros_like(target))
    zt = tl.load(logits_ptr + row * stride0 + safe_target * stride1)
    log_s = tl.log(s)
    lse = mx + log_s

    if LABEL_SMOOTHING == 0.0:
        nll = -zt + lse
        if HAS_WEIGHT:
            w = tl.load(weight_ptr + safe_target)
            loss = nll * w
            denom = w
        else:
            loss = nll
            denom = 1.0
    else:
        log_prob = tl.where(col_mask, x - lse, 0.0)
        smooth_loss = -tl.sum(log_prob, axis=0) / C
        nll = -zt + lse
        if HAS_WEIGHT:
            w = tl.load(weight_ptr + safe_target)
            loss = (1.0 - LABEL_SMOOTHING) * nll * w + LABEL_SMOOTHING * smooth_loss
            denom = w
        else:
            loss = (1.0 - LABEL_SMOOTHING) * nll + LABEL_SMOOTHING * smooth_loss
            denom = 1.0

    tl.store(row_loss_ptr + row, tl.where(valid, loss, 0.0))
    if HAS_WEIGHT:
        tl.store(row_denom_ptr + row, tl.where(valid, denom, 0.0))


@triton.jit
def finalize_loss_kernel(
    row_loss_ptr,
    row_denom_ptr,
    targets_ptr,
    N,
    ignore_index,
    target_stride,
    REDUCE_MEAN: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    t = tl.load(targets_ptr + offs * target_stride, mask=mask, other=ignore_index)
    loss = tl.load(row_loss_ptr + offs, mask=mask, other=0.0)
    ignore = ignore_index.to(tl.int64)
    valid = mask & (t != ignore)

    total = tl.sum(tl.where(valid, loss, 0.0), axis=0)
    if HAS_WEIGHT:
        denom_vals = tl.load(row_denom_ptr + offs, mask=mask, other=0.0)
        denom = tl.sum(tl.where(valid, denom_vals, 0.0), axis=0)
    else:
        denom = tl.sum(tl.where(valid, 1.0, 0.0), axis=0)

    if REDUCE_MEAN:
        result = tl.where(denom > 0.0, total / denom, 0.0)
    else:
        result = total

    tl.store(row_loss_ptr, result)


def build(context):
    def fused_cross_entropy_softmax_layernorm(
        logits,
        targets,
        normalized_shape,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        eps=1e-5,
        *,
        out=None,
    ):
        if logits.dim() != 2 or targets.dim() != 1:
            raise NotImplementedError(
                "fused_cross_entropy_softmax_layernorm supports 2D logits "
                "and 1D class-index targets for this profile"
            )
        if logits.dtype != torch.float32 or targets.dtype != torch.int64:
            raise NotImplementedError("profile requires float32 logits and int64 targets")
        if not logits.is_cuda or not targets.is_cuda:
            raise NotImplementedError("inputs must be CUDA tensors")

        N, C = logits.shape
        if isinstance(normalized_shape, int):
            norm_size = normalized_shape
        elif hasattr(normalized_shape, "__iter__"):
            norm_shape = list(normalized_shape)
            if len(norm_shape) != 1:
                raise NotImplementedError("only single-dimension normalized_shape is supported")
            norm_size = int(norm_shape[0])
        else:
            norm_size = int(normalized_shape)
        if norm_size != C:
            raise NotImplementedError("normalized_shape must equal the class dimension")

        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        if out is None:
            out = torch.empty_like(logits)
        elif out.dtype != logits.dtype or out.shape != logits.shape:
            raise ValueError("out must have the same dtype and shape as logits")

        has_weight = weight is not None
        if has_weight:
            if weight.dim() != 1 or weight.shape[0] != C:
                raise ValueError("weight must have size C")
            if weight.dtype != torch.float32 or not weight.is_cuda:
                raise NotImplementedError("profile requires a CUDA float32 class weight")

        row_loss = torch.empty((N,), device=logits.device, dtype=torch.float32)
        if has_weight:
            row_denom = torch.empty((N,), device=logits.device, dtype=torch.float32)
        else:
            row_denom = row_loss

        fused_row_kernel[(N,)](
            logits,
            targets,
            row_loss,
            out,
            weight if has_weight else logits,
            row_denom,
            N,
            C,
            logits.stride(0),
            logits.stride(1),
            out.stride(0),
            out.stride(1),
            targets.stride(0),
            ignore_index,
            eps,
            HAS_WEIGHT=has_weight,
            LABEL_SMOOTHING=label_smoothing,
        )

        if reduction != "none":
            block_n = 1 << (N - 1).bit_length()
            finalize_loss_kernel[(1,)](
                row_loss,
                row_denom,
                targets,
                N,
                ignore_index,
                targets.stride(0),
                REDUCE_MEAN=(reduction == "mean"),
                HAS_WEIGHT=has_weight,
                BLOCK_N=block_n,
                num_warps=8,
            )
            return row_loss[0], out

        return row_loss, out

    return fused_cross_entropy_softmax_layernorm
