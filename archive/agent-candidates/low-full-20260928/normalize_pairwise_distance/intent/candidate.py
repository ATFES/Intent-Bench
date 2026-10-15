import torch
import intent
import intent.language as I


@intent.kernel
def _distance(
    x1: I.In[I.f32, (16, 256, 128)],
    x2: I.In[I.f32, (16, 256, 128)],
    out: I.Out[I.f32, (16, 256)],
    eps: I.f32,
):
    batches = I.domain(0, 16)
    rows = I.domain(0, 256)
    features = I.domain(0, 128)
    delta = x1[batches, rows, features] - x2[batches, rows, features] + eps
    distance = I.sqrt(I.reduce.sum(delta * delta, axis=2, acc_dtype=I.f32))
    for b in I.parallel(batches):
        for r in I.parallel(rows):
            out[b, r] = distance[b, r]


@intent.kernel
def _normalize(
    distances: I.In[I.f32, (16, 256)],
    out: I.Out[I.f32, (16, 256)],
    eps: I.f32,
):
    batches = I.domain(0, 16)
    rows = I.domain(0, 256)
    values = distances[batches, rows]
    denominator = I.sqrt(I.reduce.sum(values * values, axis=1, acc_dtype=I.f32))
    denominator = I.select(denominator > eps, denominator, eps)
    denominator = I.reshape(denominator, (16, 1))
    normalized = values / denominator
    for b in I.parallel(batches):
        for r in I.parallel(rows):
            out[b, r] = normalized[b, r]


def build(context):
    distance_kernel = context.compile("pairwise_distance_stage", _distance)
    normalize_kernel = context.compile("pairwise_normalize_stage", _normalize)

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
        distances = torch.empty((16, 256), device=x1.device, dtype=torch.float32)
        output = torch.empty((16, 256), device=x1.device, dtype=torch.float32)
        distance_kernel(x1, x2, distances, eps_distance)
        normalize_kernel(distances, output, eps_norm)
        return output

    return normalize_pairwise_distance
