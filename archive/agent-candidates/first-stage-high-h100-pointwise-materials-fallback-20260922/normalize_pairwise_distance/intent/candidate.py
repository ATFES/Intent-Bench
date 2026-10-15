import torch
import intent
import intent.language as I


@intent.kernel
def _pairwise_distance_kernel(
    x1: I.In[I.f32, ("M", "N", "K")],
    x2: I.In[I.f32, ("M", "N", "K")],
    distances: I.Out[I.f32, ("M", "N")],
    p_distance: I.f32,
    eps_distance: I.f32,
):
    M, N, K = x1.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    features = I.domain(0, K)

    delta = x1[rows, columns, features] - x2[rows, columns, features]
    # torch.pairwise_distance applies eps to every coordinate before the norm.
    shifted = delta + eps_distance

    two = I.cast(2.0, I.f32)
    if p_distance == two:
        powers = shifted * shifted
        summed = I.reduce.sum(powers, axis=2, acc_dtype=I.f32)
        distance = I.sqrt(summed)
    else:
        magnitude = I.abs(shifted)
        powers = I.exp(I.log(magnitude) * p_distance)
        summed = I.reduce.sum(powers, axis=2, acc_dtype=I.f32)
        distance = I.exp(I.fdiv(I.log(summed), p_distance))

    distances[rows, columns] = distance


@intent.kernel
def _normalize_distance_kernel(
    distances: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
    p_norm: I.f32,
    eps_norm: I.f32,
):
    M, N = distances.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    two = I.cast(2.0, I.f32)
    if p_norm == two:
        powers = distances * distances
        summed = I.reduce.sum(powers, axis=1, acc_dtype=I.f32)
        norm = I.sqrt(summed)
    else:
        powers = I.exp(I.log(I.abs(distances)) * p_norm)
        summed = I.reduce.sum(powers, axis=1, acc_dtype=I.f32)
        norm = I.exp(I.fdiv(I.log(summed), p_norm))

    denominator = I.select(norm > eps_norm, norm, eps_norm)
    denominator = I.reshape(denominator, (M, 1))
    normalized = I.fdiv(distances[rows, columns], denominator)
    output[rows, columns] = normalized


def build(context):
    distance_artifact = context.compile(
        "normalize_pairwise_distance_distance",
        _pairwise_distance_kernel,
        constexprs={},
    )
    normalize_artifact = context.compile(
        "normalize_pairwise_distance_normalize",
        _normalize_distance_kernel,
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
        M, N, _ = x1.shape
        distances = torch.empty((M, N), dtype=x1.dtype, device=x1.device)
        output = torch.empty((M, N), dtype=x1.dtype, device=x1.device)

        distance_artifact(
            x1,
            x2,
            distances,
            float(p_distance),
            float(eps_distance),
        )
        normalize_artifact(
            distances,
            output,
            float(p_norm),
            float(eps_norm),
        )
        return output

    return normalize_pairwise_distance
