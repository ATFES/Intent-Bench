import torch
import triton
import triton.language as tl


@triton.jit
def _conv_selu_stats(x, w, b, y, sums, sqs, H, W, OH, OW, C_in, C_out,
                     eps, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C_out
    co = pid % C_out
    offs = tl.arange(0, BLOCK)
    total = OH * OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for base in tl.range(0, total, BLOCK):
        q = base + offs
        oh = q // OW
        ow = q - oh * OW
        valid = q < total
        v = tl.zeros((BLOCK,), dtype=tl.float32)
        for ci in tl.range(0, C_in):
            for kh in tl.range(0, 3):
                for kw in tl.range(0, 3):
                    iy = oh + kh
                    ix = ow + kw
                    xv = tl.load(x + ((n * C_in + ci) * H + iy) * W + ix,
                                 mask=valid, other=0.0)
                    wv = tl.load(w + ((co * C_in + ci) * 3 + kh) * 3 + kw)
                    v += xv * wv
        if HAS_BIAS:
            v += tl.load(b + co)
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        v = tl.where(v > 0, v, alpha * tl.exp(v) - alpha) * scale
        tl.store(y + (n * C_out + co) * total + q, v, mask=valid)
        acc += tl.where(valid, v, 0.0)
    mean = tl.sum(acc, axis=0) / total
    var_acc = tl.zeros((), dtype=tl.float32)
    for base in tl.range(0, total, BLOCK):
        q = base + offs
        valid = q < total
        z = tl.load(y + (n * C_out + co) * total + q, mask=valid, other=0.0)
        var_acc += tl.sum(tl.where(valid, (z - mean) * (z - mean), 0.0), axis=0)
    tl.store(sums + n * C_out + co, mean)
    tl.store(sqs + n * C_out + co, var_acc / total)


@triton.jit
def _normalize(y, sums, sqs, out, total, C_out, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C_out
    co = pid % C_out
    offs = tl.arange(0, BLOCK)
    mean = tl.load(sums + n * C_out + co)
    var = tl.load(sqs + n * C_out + co)
    for base in tl.range(0, total, BLOCK):
        q = base + offs
        mask = q < total
        z = tl.load(y + (n * C_out + co) * total + q, mask=mask, other=0.0)
        z = (z - mean) * tl.rsqrt(var + eps)
        tl.store(out + (n * C_out + co) * total + q, z, mask=mask)


def build(context):
    def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1,
                                        padding=0, dilation=1, groups=1,
                                        num_features=None, eps=1e-5,
                                        momentum=0.1, affine=False,
                                        track_running_stats=False):
        n, cin, h, w = input.shape
        cout, _, kh, kw = weight.shape
        oh = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        ow = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
        y = torch.empty((n, cout, oh, ow), device=input.device, dtype=input.dtype)
        out = torch.empty_like(y)
        sums = torch.empty((n, cout), device=input.device, dtype=torch.float32)
        sqs = torch.empty_like(sums)
        if bias is None:
            bias = torch.empty((cout,), device=input.device, dtype=input.dtype)
            has_bias = False
        else:
            has_bias = True
        _conv_selu_stats[(n * cout,)](input, weight, bias, y, sums, sqs, h, w,
                                      oh, ow, cin, cout, eps,
                                      HAS_BIAS=has_bias, BLOCK=128)
        _normalize[(n * cout,)](y, sums, sqs, out, oh * ow, cout, eps, BLOCK=128)
        return out

    return fused_instance_norm_selu_conv2d
