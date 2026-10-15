import torch
import triton
import triton.language as tl


@triton.jit
def _conv_sigmoid_bias(inp, weight, bias, out, OH, OW,
                       IH: tl.constexpr, IW: tl.constexpr,
                       IC: tl.constexpr, OC: tl.constexpr,
                       KH: tl.constexpr, KW: tl.constexpr,
                       BO: tl.constexpr, BP: tl.constexpr):
    pid = tl.program_id(0)
    po = pid * BO + tl.arange(0, BO)
    pp = tl.program_id(1) * BP + tl.arange(0, BP)
    om = (po < OC)[:, None]
    pm = pp < OH * OW
    oh = pp // OW
    ow = pp - oh * OW
    acc = tl.zeros((BO, BP), dtype=tl.float32)
    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                xi = oh + kh
                xj = ow + kw
                xv = tl.load(inp + ic * IH * IW + xi * IW + xj, mask=pm, other=0.0)
                w = tl.load(weight + po * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw,
                            mask=om[:, 0], other=0.0)
                acc += w[:, None] * xv[None, :]
    b = tl.load(bias + po, mask=om[:, 0], other=0.0)
    acc += b[:, None]
    result = 1.0 / (1.0 + tl.exp(-acc))
    out_off = po[:, None] * (OH * OW) + pp[None, :]
    tl.store(out + out_off, result, mask=om & pm[None, :])


@triton.jit
def _conv_sigmoid_nobias(inp, weight, out, OH, OW,
                         IH: tl.constexpr, IW: tl.constexpr,
                         IC: tl.constexpr, OC: tl.constexpr,
                         KH: tl.constexpr, KW: tl.constexpr,
                         BO: tl.constexpr, BP: tl.constexpr):
    pid = tl.program_id(0)
    po = pid * BO + tl.arange(0, BO)
    pp = tl.program_id(1) * BP + tl.arange(0, BP)
    om = (po < OC)[:, None]
    pm = pp < OH * OW
    oh = pp // OW
    ow = pp - oh * OW
    acc = tl.zeros((BO, BP), dtype=tl.float32)
    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                xv = tl.load(inp + ic * IH * IW + (oh + kh) * IW + (ow + kw),
                             mask=pm, other=0.0)
                w = tl.load(weight + po * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw,
                            mask=om[:, 0], other=0.0)
                acc += w[:, None] * xv[None, :]
    result = 1.0 / (1.0 + tl.exp(-acc))
    out_off = po[:, None] * (OH * OW) + pp[None, :]
    tl.store(out + out_off, result, mask=om & pm[None, :])


def build(context):
    def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0,
                       dilation=1, groups=1, out=None):
        if (input.ndim != 4 or weight.ndim != 4 or input.shape[0] != 1 or
                input.shape[1] != 64 or weight.shape != (128, 64, 3, 3) or
                input.shape[2:] != (64, 64) or stride != 1 or padding != 0 or
                dilation != 1 or groups != 1):
            raise ValueError("unsupported convolution configuration")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != 128):
            raise ValueError("unsupported bias shape")
        result = out if out is not None else torch.empty((1, 128, 62, 62),
                                                          device=input.device,
                                                          dtype=input.dtype)
        if result.shape != (1, 128, 62, 62) or result.dtype != input.dtype:
            raise ValueError("invalid out tensor")
        grid = (triton.cdiv(128, 8), triton.cdiv(62 * 62, 32))
        if bias is None:
            _conv_sigmoid_nobias[grid](input, weight, result, 62, 62,
                                       IH=64, IW=64, IC=64, OC=128,
                                       KH=3, KW=3, BO=8, BP=32)
        else:
            _conv_sigmoid_bias[grid](input, weight, bias, result, 62, 62,
                                     IH=64, IW=64, IC=64, OC=128,
                                     KH=3, KW=3, BO=8, BP=32)
        return result

    return sigmoid_conv2d
