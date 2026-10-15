import torch
import intent
import intent.language as I


@intent.kernel
def pairwise_distance_kernel(
    x1: I.In[I.f32, ("B", "M", "K")],
    x2: I.In[I.f32, ("B", "M", "K")],
    dist: I.Out[I.f32, ("B", "M")],
    eps_distance: I.f32,
):
    B, M, K = x1.shape
    rows = I.domain(0, B)
    cols = I.domain(0, M)
    ks = I.domain(0, K)

    lhs = x1[rows, cols, ks]
    rhs = x2[rows, cols, ks]
    shifted = lhs - rhs + eps_distance
    squared = shifted * shifted
    distance = I.sqrt(I.reduce.sum(squared, axis=2))

    dist[rows, cols] = distance


@intent.kernel
def normalize_pairwise_kernel(
    dist: I.In[I.f32, ("B", "M")],
    out: I.Out[I.f32, ("B", "M")],
    eps_norm: I.f32,
):
    B, M = dist.shape
    rows = I.domain(0, B)
    cols = I.domain(0, M)

    distance = dist[rows, cols]
    norm = I.sqrt(I.reduce.sum(distance * distance, axis=1))
    scale = I.maximum(norm, eps_norm)
    scale_2d = I.reshape(scale, (B, 1))

    out[rows, cols] = distance / scale_2d


def build(context):
    pairwise_artifact = context.compile(
        "normalize_pairwise_distance_pairwise",
        pairwise_distance_kernel,
        constexprs={},
    )
    normalize_artifact = context.compile(
        "normalize_pairwise_distance_normalize",
        normalize_pairwise_kernel,
        constexprs={},
    )

    def normalize_pairwise_distance(
        x1,
        x2,
        p_distance=2.0,
        eps_distance=1e-6,
        keepdim=False,
        p_norm=2,
        dim_norm=1,
        eps_norm=1e-12,
    ):
        B, M, _ = x1.shape
        dist = torch.empty((B, M), device=x1.device, dtype=x1.dtype)
        out = torch.empty((B, M), device=x1.device, dtype=x1.dtype)

        pairwise_artifact(x1, x2, dist, eps_distance)
        normalize_artifact(dist, out, eps_norm)

        return out

    return normalize_pairwise_distance
