import torch
import triton
import triton.language as tl


@triton.jit
def _conv_add_kernel(
    x, w, bias, other, out,
    total_p, OC, IC, IH, IW, KH, KW, OH, OW,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
    alpha, HAS_BIAS: tl.constexpr, OTHER_TENSOR: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    oc = tl.program_id(1) * BLOCK_OC + tl.arange(0, BLOCK_OC)
    pm = p < total_p
    om = oc < OC
    n = p // (OH * OW)
    rem = p - n * (OH * OW)
    oh = rem // OW
    ow = rem - (oh * OW)
    g = oc // (OC // groups)
    ic_per_g = IC // groups
    ic0 = g * ic_per_g
    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)
    for r in range(ic_per_g * KH * KW):
        t = r // (KH * KW)
        q = r - t * (KH * KW)
        kh = q // KW
        kw = q - kh * KW
        iy = oh * stride_h - pad_h + kh * dil_h
        ix = ow * stride_w - pad_w + kw * dil_w
        xm = pm[:, None] & (iy[:, None] >= 0) & (iy[:, None] < IH) & (ix[:, None] >= 0) & (ix[:, None] < IW)
        xx = tl.load(x + ((n[:, None] * IC + (ic0[None, :] + t)) * IH + iy[:, None]) * IW + ix[:, None], mask=xm, other=0.0)
        ww = tl.load(w + ((oc[None, :] * ic_per_g + t) * KH + kh) * KW + kw, mask=om[None, :], other=0.0)
        acc += xx * ww
    if HAS_BIAS:
        bb = tl.load(bias + oc, mask=om, other=0.0)
        acc += bb[None, :]
    if OTHER_TENSOR:
        oo = tl.load(other + ((n[:, None] * OC + oc[None, :]) * OH + oh[:, None]) * OW + ow[:, None],
                     mask=pm[:, None] & om[None, :], other=0.0)
        acc += alpha * oo
    else:
        acc += alpha
    tl.store(out + p[:, None] * OC + oc[None, :], acc, mask=pm[:, None] & om[None, :])


def _pair(v):
    return (v, v) if isinstance(v, int) else (v[0], v[1])


def build(context):
    def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0,
                   dilation=1, groups=1, alpha=1, out=None):
        sh, sw = _pair(stride)
        dh, dw = _pair(dilation)
        kh, kw = weight.shape[-2:]
        if padding == "same":
            ph = ((input.shape[-2] - 1) * sh + (kh - 1) * dh + 1 - input.shape[-2]) // 2
            pw = ((input.shape[-1] - 1) * sw + (kw - 1) * dw + 1 - input.shape[-1]) // 2
        elif padding == "valid":
            ph, pw = 0, 0
        else:
            ph, pw = _pair(padding)
        n, ic, ih, iw = input.shape
        oc = weight.shape[0]
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        if out is None:
            out = torch.empty((n, oc, oh, ow), device=input.device, dtype=input.dtype)
        # The kernel uses a flattened NHWC-like logical output indexing while
        # reading/writing contiguous NCHW tensors through an equivalent layout.
        # Convert the output indexing to NCHW by using a contiguous temporary view.
        target = out
        if not target.is_contiguous():
            target = torch.empty((n, oc, oh, ow), device=input.device, dtype=input.dtype)
        x = input.contiguous()
        w = weight.contiguous()
        if bias is None:
            b = torch.empty((1,), device=input.device, dtype=input.dtype)
        else:
            b = bias.contiguous()
        if isinstance(other, torch.Tensor):
            add = other.contiguous()
            other_tensor = True
        elif other is None:
            add = torch.empty((1,), device=input.device, dtype=input.dtype)
            other_tensor = False
            alpha = 0
        else:
            add = torch.empty((1,), device=input.device, dtype=input.dtype)
            other_tensor = False
            alpha = alpha * other
        # Kernel's flattened output is [N, OH, OW, OC]; write through a
        # contiguous temporary and permute into the requested NCHW result.
        flat = torch.empty((n * oh * ow, oc), device=input.device, dtype=input.dtype)
        grid = (triton.cdiv(n * oh * ow, 128), triton.cdiv(oc, 32))
        _conv_add_kernel[grid](x, w, b, add, flat, n * oh * ow, oc, ic, ih, iw,
                               kh, kw, oh, ow, sh, sw, ph, pw, dh, dw, groups,
                               alpha, HAS_BIAS=bias is not None,
                               OTHER_TENSOR=other_tensor, BLOCK_P=128, BLOCK_OC=32)
        result = flat.reshape(n, oh, ow, oc).permute(0, 3, 1, 2)
        if target is out:
            out.copy_(result)
            return out
        target.copy_(result)
        out.copy_(target)
        return out
    return conv2d_add
