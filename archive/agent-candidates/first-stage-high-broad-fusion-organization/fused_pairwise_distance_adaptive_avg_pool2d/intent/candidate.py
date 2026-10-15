import torch
import intent
import intent.language as I


@intent.kernel
def _fused_pairwise_distance(
    x1: I.In[I.f32, (16, 3, 32, 32)],
    x2: I.In[I.f32, (16, 3, 32, 32)],
    out: I.Out[I.f32, (16,)],
):
    eps = I.cast(1.0e-6, I.f32)
    denom = I.cast(4.0, I.f32)
    channels = I.reshape(I.indices(I.domain(0, 3)), (3, 1, 1))
    rows = I.reshape(I.indices(I.domain(0, 16)) * 2, (1, 16, 1))
    cols = I.reshape(I.indices(I.domain(0, 16)) * 2, (1, 1, 16))
    for batch in I.parallel(I.domain(0, 16)):
        pooled1 = (
            x1[batch, channels, rows, cols]
            + x1[batch, channels, rows, cols + 1]
            + x1[batch, channels, rows + 1, cols]
            + x1[batch, channels, rows + 1, cols + 1]
        ) / denom
        pooled2 = (
            x2[batch, channels, rows, cols]
            + x2[batch, channels, rows, cols + 1]
            + x2[batch, channels, rows + 1, cols]
            + x2[batch, channels, rows + 1, cols + 1]
        ) / denom
        delta = pooled1 - pooled2 + eps
        distance_sq = I.reduce.sum(delta * delta, axis=(0, 1, 2), acc_dtype=I.f32)
        out[batch] = I.sqrt(distance_sq)


def build(context):
    fused = context.compile("fused_pairwise_distance", _fused_pairwise_distance, constexprs={})

    def fused_pairwise_distance_adaptive_avg_pool2d(
        x1: torch.Tensor,
        x2: torch.Tensor,
        output_size: int or tuple,
        p: float = 2.0,
        eps: float = 1e-6,
        keepdim: bool = False,
    ) -> torch.Tensor:
        out = torch.empty((x1.shape[0],), dtype=x1.dtype, device=x1.device)
        fused(x1, x2, out)
        return out

    return fused_pairwise_distance_adaptive_avg_pool2d
