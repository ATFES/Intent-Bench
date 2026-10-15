from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def euclidean_pairwise_distance(
    x1: I.In[I.f32, ("B", "N", "D")],
    x2: I.In[I.f32, ("B", "N", "D")],
    distances: I.Out[I.f32, ("B", "N")],
    eps_distance: I.f32,
) -> None:
    batches = I.domain(0, x1.shape[0])
    rows = I.domain(0, x1.shape[1])
    features = I.domain(0, x1.shape[2])
    differences = x1[batches, rows, features] - x2[batches, rows, features]
    squared_distances = I.reduce.sum(
        differences * differences,
        axis=2,
        acc_dtype=I.f32,
    )
    distances[batches, rows] = I.sqrt(squared_distances) + eps_distance


@intent.kernel
def normalize_rows_l2(
    distances: I.In[I.f32, ("B", "N")],
    result: I.Out[I.f32, ("B", "N")],
    eps_norm: I.f32,
) -> None:
    batches = I.domain(0, distances.shape[0])
    rows = I.domain(0, distances.shape[1])
    values = distances[batches, rows]
    squared_norms = I.reduce.sum(
        values * values,
        axis=1,
        acc_dtype=I.f32,
    )
    normalizers = I.maximum(I.sqrt(squared_norms), eps_norm)
    result[batches, rows] = values / I.reshape(
        normalizers,
        (distances.shape[0], 1),
    )


def build(context):
    distance_kernel = context.compile(
        "normalize_pairwise_distance_euclidean",
        euclidean_pairwise_distance,
    )
    normalize_kernel = context.compile(
        "normalize_pairwise_distance_l2_rows",
        normalize_rows_l2,
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
        distances = torch.empty(
            (x1.shape[0], x1.shape[1]),
            device=x1.device,
            dtype=x1.dtype,
        )
        result = torch.empty_like(distances)
        distance_kernel(x1, x2, distances, eps_distance)
        normalize_kernel(distances, result, eps_norm)
        return result

    return normalize_pairwise_distance
