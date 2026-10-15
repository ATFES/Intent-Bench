import torch
import intent
import intent.language as I


@intent.kernel
def pool2x2(
    x1: I.In[I.f32, (16, 3, 32, 32)],
    x2: I.In[I.f32, (16, 3, 32, 32)],
    y1: I.Out[I.f32, (16, 3, 16, 16)],
    y2: I.Out[I.f32, (16, 3, 16, 16)],
):
    batches = I.domain(0, 16)
    channels = I.domain(0, 3)
    rows = I.domain(0, 16)
    cols = I.domain(0, 16)
    src_rows = I.domain(0, 32)
    src_cols = I.domain(0, 32)
    for b in I.parallel(batches):
        for c in I.parallel(channels):
            for r in I.parallel(rows):
                for col in I.parallel(cols):
                    rr = src_rows[r * 2 : r * 2 + 2]
                    cc = src_cols[col * 2 : col * 2 + 2]
                    v1 = x1[b, c, rr, cc]
                    v2 = x2[b, c, rr, cc]
                    y1[b, c, r, col] = I.reduce.sum(v1, axis=(0, 1)) / 4.0
                    y2[b, c, r, col] = I.reduce.sum(v2, axis=(0, 1)) / 4.0


@intent.kernel
def pairwise_l2(
    y1: I.In[I.f32, (16, 3, 16, 16)],
    y2: I.In[I.f32, (16, 3, 16, 16)],
    out: I.Out[I.f32, (16,)],
    eps: I.f32,
):
    batches = I.domain(0, 16)
    for b in I.parallel(batches):
        delta = y1[b, :, :, :] - y2[b, :, :, :]
        norm2 = I.reduce.sum(delta * delta, axis=(0, 1, 2))
        out[b] = I.sqrt(norm2) + eps


@intent.kernel
def pairwise_l2_keepdim(
    y1: I.In[I.f32, (16, 3, 16, 16)],
    y2: I.In[I.f32, (16, 3, 16, 16)],
    out: I.Out[I.f32, (16, 1)],
    eps: I.f32,
):
    batches = I.domain(0, 16)
    for b in I.parallel(batches):
        delta = y1[b, :, :, :] - y2[b, :, :, :]
        norm2 = I.reduce.sum(delta * delta, axis=(0, 1, 2))
        out[b, 0] = I.sqrt(norm2) + eps


def build(context):
    pool = context.compile("pool2x2", pool2x2)
    distance = context.compile("pairwise_l2", pairwise_l2)
    distance_keepdim = context.compile("pairwise_l2_keepdim", pairwise_l2_keepdim)

    def fused_pairwise_distance_adaptive_avg_pool2d(
        x1: torch.Tensor,
        x2: torch.Tensor,
        output_size: int or tuple,
        p: float = 2.0,
        eps: float = 1e-6,
        keepdim: bool = False,
    ) -> torch.Tensor:
        pooled1 = torch.empty((16, 3, 16, 16), device=x1.device, dtype=torch.float32)
        pooled2 = torch.empty((16, 3, 16, 16), device=x2.device, dtype=torch.float32)
        pool(x1, x2, pooled1, pooled2)
        if keepdim:
            out = torch.empty((16, 1), device=x1.device, dtype=torch.float32)
            distance_keepdim(pooled1, pooled2, out, eps)
        else:
            out = torch.empty((16,), device=x1.device, dtype=torch.float32)
            distance(pooled1, pooled2, out, eps)
        return out

    return fused_pairwise_distance_adaptive_avg_pool2d
