import torch
import triton
import triton.language as tl


@triton.jit
def _fused_avg_pool2d_cosine_similarity(
    x1_ptr,
    x2_ptr,
    out_ptr,
    n_batches,
    n_channels,
    height,
    width,
    out_height,
    out_width,
    BLOCK_C: tl.constexpr,
    KERNEL: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % out_width
    t = pid // out_width
    oh = t % out_height
    n = t // out_height

    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < n_channels
    channel_stride = height * width
    batch_stride = n_channels * channel_stride

    pooled = tl.zeros((), dtype=tl.float32)
    for kh in range(KERNEL):
        ih = oh * STRIDE - PADDING + kh
        h_valid = (ih >= 0) & (ih < height)
        for kw in range(KERNEL):
            iw = ow * STRIDE - PADDING + kw
            valid = h_valid & (iw >= 0) & (iw < width)

            base = n * batch_stride + ih * width + iw
            offsets = base + channels * channel_stride
            a = tl.load(x1_ptr + offsets, mask=channel_mask & valid, other=0.0)
            b = tl.load(x2_ptr + offsets, mask=channel_mask & valid, other=0.0)

            dot = tl.sum(a * b, axis=0)
            norm_a = tl.sqrt(tl.sum(a * a, axis=0))
            norm_b = tl.sqrt(tl.sum(b * b, axis=0))
            denom = tl.maximum(norm_a, EPS) * tl.maximum(norm_b, EPS)
            similarity = dot / denom
            pooled += tl.where(valid, similarity, 0.0)

    output_offset = n * out_height * out_width + oh * out_width + ow
    tl.store(out_ptr + output_offset, pooled / (KERNEL * KERNEL))


def build(context):
    def wrapper(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
        if stride is None:
            stride = kernel_size

        n_batches, n_channels, height, width = x1.shape
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        output = torch.empty(
            (n_batches, 1, out_height, out_width),
            device=x1.device,
            dtype=x1.dtype,
        )

        grid = (n_batches * out_height * out_width,)
        _fused_avg_pool2d_cosine_similarity[grid](
            x1,
            x2,
            output,
            n_batches,
            n_channels,
            height,
            width,
            out_height,
            out_width,
            BLOCK_C=64,
            KERNEL=kernel_size,
            STRIDE=stride,
            PADDING=padding,
            EPS=eps,
        )
        return output

    return wrapper
