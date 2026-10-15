from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['height', 'width'],
)
@triton.jit
def _conv_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch,
    height,
    width,
    out_height,
    out_width,
    out_channels,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    spatial = out_height * out_width
    m_valid = m < batch * spatial
    n_valid = n < out_channels

    b = m // spatial
    p = m % spatial
    oh = p // out_width
    ow = p % out_width

    k = tl.arange(0, 32)
    k_valid = k < 27
    ic = k // 9
    kk = k % 9
    kh = kk // 3
    kw = kk % 3

    ih = oh[:, None] * stride_h + kh[None, :] - pad_h
    iw = ow[:, None] * stride_w + kw[None, :] - pad_w
    x_offsets = b[:, None] * (3 * height * width) + ic[None, :] * (height * width)
    x_offsets += ih * width + iw
    x_mask = m_valid[:, None] & k_valid[None, :]
    x_mask &= (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
    x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    w_offsets = n[None, :] * 27 + k[:, None]
    w_mask = n_valid[None, :] & k_valid[:, None]
    w = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

    acc = tl.dot(x, w, input_precision='ieee')
    if HAS_BIAS:
        bias = tl.load(b_ptr + n, mask=n_valid, other=0.0)
        acc += bias[None, :]
    acc = tl.maximum(acc, 0.0)

    out_offsets = b[:, None] * (out_channels * spatial)
    out_offsets += n[None, :] * spatial + p[:, None]
    tl.store(out_ptr + out_offsets, acc, mask=m_valid[:, None] & n_valid[None, :])


def _pair(value):
    if isinstance(value, (tuple, list)):
        return value[0], value[1]
    return value, value


def build(context):
    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
        stride_h, stride_w = _pair(stride)
        dilation_h, dilation_w = _pair(dilation)
        if isinstance(padding, str):
            pad_h, pad_w = (1, 1) if padding == 'same' else (0, 0)
        else:
            pad_h, pad_w = _pair(padding)

        batch, in_channels, height, width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        output = torch.empty(
            (batch, out_channels, out_height, out_width),
            device=input.device,
            dtype=input.dtype,
        )

        grid = lambda meta: (
            triton.cdiv(batch * out_height * out_width, meta['BLOCK_M']),
            triton.cdiv(out_channels, meta['BLOCK_N']),
        )
        bias_ptr = weight if bias is None else bias
        _conv_relu_kernel[grid](
            input,
            weight,
            bias_ptr,
            output,
            batch,
            height,
            width,
            out_height,
            out_width,
            out_channels,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            HAS_BIAS=bias is not None,
        )
        return output

    return wrapper
