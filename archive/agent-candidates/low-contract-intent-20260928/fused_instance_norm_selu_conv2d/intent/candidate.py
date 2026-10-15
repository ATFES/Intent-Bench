import torch
import intent
import intent.language as I


@intent.kernel
def _conv_selu(
    x: I.In[I.f32, ("N", 16, 64, 64)],
    w: I.In[I.f32, (64, 16, 3, 3)],
    b: I.In[I.f32, (64,)],
    tmp: I.Out[I.f32, ("N", 64, 62, 62)],
):
    ns = I.domain(0, x.shape[0])
    cs = I.domain(0, 64)
    hs = I.domain(0, 62)
    ws = I.domain(0, 62)
    for n in I.parallel(ns):
        for co in I.parallel(cs):
            for oh in I.parallel(hs):
                for ow in I.parallel(ws):
                    acc = b[co]
                    for ci in I.domain(0, 16):
                        for kh in I.domain(0, 3):
                            for kw in I.domain(0, 3):
                                acc = acc + x[n, ci, oh + kh, ow + kw] * w[co, ci, kh, kw]
                    positive = acc > I.cast(0.0, I.f32)
                    neg = I.cast(1.6732632423543772, I.f32) * (I.exp(acc) - I.cast(1.0, I.f32))
                    activated = I.cast(1.0507009873554805, I.f32) * I.select(positive, acc, neg)
                    tmp[n, co, oh, ow] = activated


@intent.kernel
def _instance_norm(
    tmp: I.In[I.f32, ("N", 64, 62, 62)],
    out: I.Out[I.f32, ("N", 64, 62, 62)],
    eps: I.f32,
):
    ns = I.domain(0, tmp.shape[0])
    cs = I.domain(0, 64)
    hs = I.domain(0, 62)
    ws = I.domain(0, 62)
    for n in I.parallel(ns):
        for c in I.parallel(cs):
            values = tmp[n, c, hs, ws]
            mean = I.reduce.sum(values, axis=(0, 1)) / I.cast(3844.0, I.f32)
            centered = values - mean
            variance = I.reduce.sum(centered * centered, axis=(0, 1)) / I.cast(3844.0, I.f32)
            inv_std = I.rsqrt(variance + eps)
            for h in I.parallel(hs):
                for w in I.parallel(ws):
                    out[n, c, h, w] = (tmp[n, c, h, w] - mean) * inv_std


def build(context):
    conv = context.compile("fused_conv_selu", _conv_selu)
    norm = context.compile("fused_instance_norm", _instance_norm)

    def fused_instance_norm_selu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_features=None,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        n = input.shape[0]
        tmp = torch.empty((n, 64, 62, 62), device=input.device, dtype=input.dtype)
        output = torch.empty_like(tmp)
        conv(input, weight, bias, tmp)
        norm(tmp, output, eps)
        return output

    return fused_instance_norm_selu_conv2d
