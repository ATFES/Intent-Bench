import torch
import intent
import intent.language as I


@intent.kernel
def _batch_norm_inference(
    x: I.In[I.f16, ("N", "C", "H", "W")],
    mean: I.In[I.f32, ("C",)],
    var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f16, ("N", "C", "H", "W")],
    eps: I.f32,
):
    ns = I.domain(0, 32)
    cs = I.domain(0, 32)
    hs = I.domain(0, 640)
    ws = I.domain(0, 128)
    for n in I.parallel(ns):
        for c in I.parallel(cs):
            m = mean[c]
            v = var[c]
            g = weight[c]
            b = bias[c]
            inv = I.fdiv(I.cast(1.0, I.f32), I.sqrt(v + eps))
            for h in I.parallel(hs):
                for w in I.parallel(ws):
                    y = (I.cast(x[n, c, h, w], I.f32) - m) * inv * g + b
                    out[n, c, h, w] = I.cast(y, I.f16)


def build(context):
    kernel = context.compile("batch_norm_inference", _batch_norm_inference)

    def batch_norm(input, running_mean, running_var, weight=None, bias=None,
                   training=False, momentum=0.1, eps=1e-5):
        out = torch.empty_like(input)
        kernel(input, running_mean, running_var, weight, bias, out, eps)
        return out

    return batch_norm
