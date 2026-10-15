import torch
import intent
import intent.language as I


@intent.kernel
def cosine_stage(
    x1: I.In[I.f32, ("N", "C", "H", "W")],
    x2: I.In[I.f32, ("N", "C", "H", "W")],
    out: I.Out[I.f32, ("N", "H", "W")],
    eps: I.f32,
):
    ns = I.domain(0, x1.shape[0])
    hs = I.domain(0, x1.shape[2])
    ws = I.domain(0, x1.shape[3])
    cs = I.domain(0, x1.shape[1])
    for n in I.parallel(ns):
        for h in I.parallel(hs):
            for w in I.parallel(ws):
                a = x1[n, cs, h, w]
                b = x2[n, cs, h, w]
                aa = I.reduce.sum(a * a, axis=0, acc_dtype=I.f32)
                bb = I.reduce.sum(b * b, axis=0, acc_dtype=I.f32)
                ab = I.reduce.sum(a * b, axis=0, acc_dtype=I.f32)
                denom = I.sqrt(aa) * I.sqrt(bb)
                denom = I.select(denom < eps, eps, denom)
                out[n, h, w] = ab / denom


@intent.kernel
def pool_stage(
    src: I.In[I.f32, ("N", "H", "W")],
    out: I.Out[I.f32, ("N", "OH", "OW")],
    kernel_size: I.i64,
    stride: I.i64,
    padding: I.i64,
):
    ns = I.domain(0, src.shape[0])
    ohs = I.domain(0, out.shape[1])
    ows = I.domain(0, out.shape[2])
    for n in I.parallel(ns):
        for oh in I.parallel(ohs):
            for ow in I.parallel(ows):
                total = I.cast(0.0, I.f32)
                for kh in range(3):
                    for kw in range(3):
                        ih = oh * stride - padding + kh
                        iw = ow * stride - padding + kw
                        valid = (ih >= 0) & (ih < src.shape[1]) & (iw >= 0) & (iw < src.shape[2])
                        value = I.gather(src[n], (ih, iw), valid=valid, fill=I.cast(0.0, I.f32))
                        total = total + I.select(valid, value, I.cast(0.0, I.f32))
                out[n, oh, ow] = total / I.cast(kernel_size * kernel_size, I.f32)


def build(context):
    cosine = context.compile("cosine_stage", cosine_stage)
    pool = context.compile("pool_stage", pool_stage)

    def fused_avg_pool2d_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size
        n, _, h, w = x1.shape
        cosine_out = torch.empty((n, h, w), device=x1.device, dtype=torch.float32)
        oh = (h + 2 * padding - kernel_size) // stride + 1
        ow = (w + 2 * padding - kernel_size) // stride + 1
        result = torch.empty((n, 1, oh, ow), device=x1.device, dtype=torch.float32)
        cosine(x1, x2, cosine_out, eps)
        pool(cosine_out, result[:, 0], kernel_size, stride, padding)
        return result

    return fused_avg_pool2d_cosine_similarity
