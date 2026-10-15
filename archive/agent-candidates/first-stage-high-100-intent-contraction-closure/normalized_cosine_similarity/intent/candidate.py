from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _normalized_cosine_kernel(
    x1: I.In[I.f32, ("M", "N")],
    x2: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, ("M",)],
    eps_similarity: I.f32,
    p_norm: I.f32,
    eps_norm: I.f32,
):
    M, N = x1.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    x1_values = x1[rows, columns]
    x2_values = x2[rows, columns]

    x1_abs = I.abs(x1_values)
    x2_abs = I.abs(x2_values)

    # Keep the common p=2 path in sqrt(sum(x*x)) form.  The other branch
    # retains the documented Lp behavior for runtime p_norm values.
    if p_norm == 2.0:
        x1_power = x1_abs * x1_abs
        x2_power = x2_abs * x2_abs
        x1_norm = I.sqrt(I.reduce.sum(x1_power, axis=1, acc_dtype=I.f32))
        x2_norm = I.sqrt(I.reduce.sum(x2_power, axis=1, acc_dtype=I.f32))
    else:
        x1_power = I.exp(I.log(x1_abs) * p_norm)
        x2_power = I.exp(I.log(x2_abs) * p_norm)
        x1_norm_sum = I.reduce.sum(x1_power, axis=1, acc_dtype=I.f32)
        x2_norm_sum = I.reduce.sum(x2_power, axis=1, acc_dtype=I.f32)
        x1_norm = I.exp(I.log(x1_norm_sum) / p_norm)
        x2_norm = I.exp(I.log(x2_norm_sum) / p_norm)

    x1_scale = I.reshape(I.maximum(x1_norm, eps_norm), (M, 1))
    x2_scale = I.reshape(I.maximum(x2_norm, eps_norm), (M, 1))
    x1_normalized = I.fdiv(x1_values, x1_scale)
    x2_normalized = I.fdiv(x2_values, x2_scale)

    product = x1_normalized * x2_normalized
    numerator = I.reduce.sum(product, axis=1, acc_dtype=I.f32)
    x1_similarity_norm = I.sqrt(
        I.reduce.sum(x1_normalized * x1_normalized, axis=1, acc_dtype=I.f32)
    )
    x2_similarity_norm = I.sqrt(
        I.reduce.sum(x2_normalized * x2_normalized, axis=1, acc_dtype=I.f32)
    )
    denominator = I.maximum(x1_similarity_norm, eps_similarity) * I.maximum(
        x2_similarity_norm, eps_similarity
    )
    out[rows] = I.fdiv(numerator, denominator)


def build(context):
    artifact = context.compile(
        "normalized_cosine_similarity_main",
        _normalized_cosine_kernel,
        constexprs={},
    )

    def normalized_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        dim: int = 1,
        eps_similarity: float = 1e-8,
        p_norm: float = 2,
        eps_norm: float = 1e-12,
    ) -> torch.Tensor:
        normalized_dim = dim if dim >= 0 else dim + x1.ndim
        x2_broadcast = x2.expand_as(x1)
        if normalized_dim == 1:
            return artifact.run(x1, x2_broadcast, eps_similarity, p_norm, eps_norm)
        if normalized_dim == 0:
            return artifact.run(
                x1.transpose(0, 1),
                x2_broadcast.transpose(0, 1),
                eps_similarity,
                p_norm,
                eps_norm,
            )
        raise ValueError("dim must identify an axis of the rank-2 input")

    return normalized_cosine_similarity
