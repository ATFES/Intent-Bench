from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _normalized_cosine_similarity_kernel(
    x1: I.In[I.f32, ("M", "N")],
    x2: I.In[I.f32, ("M", "N")],
    result: I.Out[I.f32, ("M",)],
    p_norm: I.f32,
    eps_similarity: I.f32,
    eps_norm: I.f32,
) -> None:
    m, n = x1.shape

    if p_norm == I.cast(2.0, I.f32):
        x1_power = x1 * x1
        x2_power = x2 * x2
    else:
        x1_power = I.exp(I.log(I.abs(x1)) * p_norm)
        x2_power = I.exp(I.log(I.abs(x2)) * p_norm)

    x1_norm = I.sqrt(
        I.reduce.sum(x1_power, axis=1, acc_dtype=I.f32)
    )
    x2_norm = I.sqrt(
        I.reduce.sum(x2_power, axis=1, acc_dtype=I.f32)
    )

    normalized_x1 = x1 / I.reshape(
        I.maximum(x1_norm, eps_norm), (m, 1)
    )
    normalized_x2 = x2 / I.reshape(
        I.maximum(x2_norm, eps_norm), (m, 1)
    )

    similarity_numerator = I.reduce.sum(
        normalized_x1 * normalized_x2, axis=1, acc_dtype=I.f32
    )
    normalized_x1_norm = I.sqrt(
        I.reduce.sum(
            normalized_x1 * normalized_x1, axis=1, acc_dtype=I.f32
        )
    )
    normalized_x2_norm = I.sqrt(
        I.reduce.sum(
            normalized_x2 * normalized_x2, axis=1, acc_dtype=I.f32
        )
    )
    similarity_denominator = I.maximum(
        normalized_x1_norm, eps_similarity
    ) * I.maximum(normalized_x2_norm, eps_similarity)

    rows = I.domain(0, m)
    result[rows] = similarity_numerator / similarity_denominator


def build(context):
    kernel = context.compile(
        "normalized_cosine_similarity_rowwise",
        _normalized_cosine_similarity_kernel,
    )

    def normalized_cosine_similarity(
        x1,
        x2,
        dim: int = 1,
        eps_similarity: float = 1e-8,
        p_norm: float = 2,
        eps_norm: float = 1e-12,
    ):
        return kernel.run(
            x1,
            x2,
            p_norm,
            eps_similarity,
            eps_norm,
        )

    return normalized_cosine_similarity
