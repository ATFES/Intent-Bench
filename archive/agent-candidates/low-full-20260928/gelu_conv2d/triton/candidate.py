import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n, ci, ih, iw, co, oh, ow,
    xsn, xsc, xsh, xsw,
    wso, wsi, wskh, wskw,
    oso, osc, osh, osw,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    HAS_BIAS: tl.constexpr, APPROX_TANH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    ids = pid * BLOCK + tl.arange(0, BLOCK)
    total = n * co * oh * ow
    mask = ids < total

    ow_i = ids % ow
    t = ids // ow
    oh_i = t % oh
    t = t // oh
    co_i = t % co
    n_i = t // co

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in range(3):
        ih_i = oh_i * stride_h - pad_h + r * dilation_h
        for s in range(3):
            iw_i = ow_i * stride_w - pad_w + s * dilation_w
            valid = (ih_i >= 0) & (ih_i < ih) & (iw_i >= 0) & (iw_i < iw)
            for c in range(16):
                xoff = n_i * xsn + c * xsc + ih_i * xsh + iw_i * xsw
                woff = co_i * wso + c * wsi + r * wskh + s * wskw
                xv = tl.load(x_ptr + xoff, mask=mask & valid, other=0.0)
                wv = tl.load(w_ptr + woff, mask=mask, other=0.0)
                acc += xv * wv

    if HAS_BIAS:
        acc += tl.load(b_ptr + co_i, mask=mask, other=0.0)

    if APPROX_TANH:
        z = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        y = 0.5 * acc * (1.0 + tl.math.tanh(z))
    else:
        y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(out_ptr + n_i * oso + co_i * osc + oh_i * osh + ow_i * osw, y, mask=mask)


def _pair(v):
    if isinstance(v, int):
        return v, v
    return int(v[0]), int(v[1])


def build(context):
    def gelu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        approximate: str = "none",
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sh, sw = _pair(stride)
        dh, dw = _pair(dilation)
        if isinstance(padding, str):
            if padding == "same":
                ph = ((input.shape[2] - 1) * sh + (weight.shape[2] - 1) * dh + 1 - input.shape[2]) // 2
                pw = ((input.shape[3] - 1) * sw + (weight.shape[3] - 1) * dw + 1 - input.shape[3]) // 2
            else:
                ph = pw = 0
        else:
            ph, pw = _pair(padding)
        n, ci, ih, iw = input.shape
        co, cpg, kh, kw = weight.shape
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        if out is None:
            out = torch.empty((n, co, oh, ow), device=input.device, dtype=input.dtype)
        if groups != 1 or ci != 16 or cpg != 16 or kh != 3 or kw != 3:
            raise NotImplementedError("fixed profile kernel supports one 16-channel 3x3 group")
        dummy = weight if bias is None else bias
        block = 128
        grid = (triton.cdiv(n * co * oh * ow, block),)
        _gelu_conv2d_kernel[grid](
            input, weight, dummy, out,
            n, ci, ih, iw, co, oh, ow,
            input.stride(0), input.stride(1), input.stride(2), input.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            sh, sw, ph, pw, dh, dw,
            HAS_BIAS=bias is not None,
            APPROX_TANH=approximate == "tanh",
            BLOCK=block,
        )
        return out

    return gelu_conv2d
