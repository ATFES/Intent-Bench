from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def _conv3x3_gelu(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    n_images,
    in_channels,
    height,
    width,
    out_channels,
    out_height,
    out_width,
    HAS_BIAS: tl.constexpr,
    APPROX_TANH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    oc = tl.arange(0, BLOCK_N)

    spatial = out_height * out_width
    image = m // spatial
    rem = m - image * spatial
    oh = rem // out_width
    ow = rem - oh * out_width

    m_mask = m < n_images * spatial
    oc_mask = oc < out_channels
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # The fixed task profile is a 3x3, stride-one convolution with one-pixel
    # zero padding.  K is laid out as (input channel, kernel row, kernel col).
    for k0 in range(0, 144, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < 144
        ic = k // 9
        kh = (k - ic * 9) // 3
        kw = k - ic * 9 - kh * 3

        ih = oh[:, None] + kh[None, :] - 1
        iw = ow[:, None] + kw[None, :] - 1
        x_offsets = (
            image[:, None] * (in_channels * height * width)
            + ic[None, :] * (height * width)
            + ih * width
            + iw
        )
        x_mask = (
            m_mask[:, None]
            & k_mask[None, :]
            & (ih >= 0)
            & (ih < height)
            & (iw >= 0)
            & (iw < width)
        )
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        w_offsets = oc[None, :] * 144 + k[:, None]
        w_mask = k_mask[:, None] & oc_mask[None, :]
        w = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)
        acc += tl.dot(x, w, input_precision="ieee")

    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc, mask=oc_mask, other=0.0)[None, :]

    if APPROX_TANH:
        x = acc
        x3 = x * x * x
        z = 0.7978845608028654 * (x + 0.044715 * x3)
        result = 0.5 * x * (1.0 + tl.tanh(z))
    else:
        result = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    y_offsets = (
        (image[:, None] * out_channels + oc[None, :]) * spatial + rem[:, None]
    )
    tl.store(y_ptr + y_offsets, result, mask=m_mask[:, None] & oc_mask[None, :])


def build(context):
    def wrapper(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate="none",
        out=None,
    ):
        # The evaluator supplies the fixed profile in TASK.md.  Keep the
        # wrapper's full public signature and honor an explicitly supplied out.
        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride
        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation_w = dilation
        if isinstance(padding, str):
            if padding == "same":
                padding_h = (weight.shape[2] - 1) * dilation_h // 2
                padding_w = (weight.shape[3] - 1) * dilation_w // 2
            else:
                padding_h = padding_w = 0
        elif isinstance(padding, tuple):
            padding_h, padding_w = padding
        else:
            padding_h = padding_w = padding

        n_images, in_channels, height, width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        if out is None:
            out = torch.empty(
                (n_images, out_channels, out_height, out_width),
                device=input.device,
                dtype=input.dtype,
            )

        # This specialized kernel is selected for the supplied fixed profile.
        # The checks make accidental use with a different convolution explicit.
        if (
            kernel_h != 3
            or kernel_w != 3
            or stride_h != 1
            or stride_w != 1
            or dilation_h != 1
            or dilation_w != 1
            or padding_h != 1
            or padding_w != 1
            or groups != 1
            or in_channels != 16
            or out_channels != 32
            or height != 64
            or width != 64
        ):
            raise ValueError("candidate is specialized for the fixed profile in TASK.md")

        grid = lambda META: (
            triton.cdiv(n_images * out_height * out_width, META["BLOCK_M"]),
        )
        _conv3x3_gelu[grid](
            input,
            weight,
            bias if bias is not None else input,
            out,
            n_images,
            in_channels,
            height,
            width,
            out_channels,
            out_height,
            out_width,
            HAS_BIAS=bias is not None,
            APPROX_TANH=approximate == "tanh",
        )
        return out

    return wrapper
