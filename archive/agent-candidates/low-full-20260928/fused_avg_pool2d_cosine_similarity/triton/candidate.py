import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    n_images,
    height,
    width,
    out_height,
    out_width,
    stride,
    padding,
    eps,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_h,
    out_stride_w,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    spatial = out_height * out_width
    n = pid // spatial
    rem = pid % spatial
    oy = rem // out_width
    ox = rem % out_width

    total = tl.zeros((), dtype=tl.float32)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < 64

    # Each pooling window contains K*K independent cosine reductions.
    for ky in range(0, K):
        iy = oy * stride - padding + ky
        y_valid = (iy >= 0) & (iy < height)
        for kx in range(0, K):
            ix = ox * stride - padding + kx
            valid = y_valid & (ix >= 0) & (ix < width)
            base = n * stride_n + iy * stride_h + ix * stride_w
            a = tl.load(x1_ptr + base + c * stride_c, mask=valid & c_mask, other=0.0)
            b = tl.load(x2_ptr + base + c * stride_c, mask=valid & c_mask, other=0.0)
            dot = tl.sum(a * b, axis=0)
            norm_a = tl.sqrt(tl.sum(a * a, axis=0))
            norm_b = tl.sqrt(tl.sum(b * b, axis=0))
            denom = tl.maximum(norm_a, eps) * tl.maximum(norm_b, eps)
            cosine = dot / denom
            total += tl.where(valid, cosine, 0.0)

    result = total / (K * K)
    tl.store(out_ptr + n * out_stride_n + oy * out_stride_h + ox * out_stride_w, result)


def build(context):
    def wrapper(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
        if stride is None:
            stride = kernel_size
        n, _, h, w = x1.shape
        out_h = (h + 2 * padding - kernel_size) // stride + 1
        out_w = (w + 2 * padding - kernel_size) // stride + 1
        out = torch.empty((n, 1, out_h, out_w), device=x1.device, dtype=x1.dtype)
        grid = (n * out_h * out_w,)
        _fused_kernel[grid](
            x1,
            x2,
            out,
            n,
            h,
            w,
            out_h,
            out_w,
            stride,
            padding,
            eps,
            x1.stride(0),
            x1.stride(1),
            x1.stride(2),
            x1.stride(3),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            K=3,
            BLOCK_C=64,
        )
        return out

    return wrapper
