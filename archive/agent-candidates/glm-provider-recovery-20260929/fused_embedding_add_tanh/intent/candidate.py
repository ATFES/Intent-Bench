from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def fused_embedding_add_tanh_default(
    input_indices: I.In[I.i64, ("M", "N")],
    weight: I.In[I.f32, ("V", "D")],
    other: I.In[I.f32, ("M", "N", "D")],
    output: I.Out[I.f32, ("M", "N", "D")],
) -> None:
    rows = input_indices[:, :]
    row_coordinates = I.reshape(rows, (input_indices.shape[0], input_indices.shape[1], 1))
    embedding_coordinates = I.reshape(
        I.indices(I.domain(0, weight.shape[1])),
        (1, 1, weight.shape[1]),
    )
    embedded = weight[row_coordinates, embedding_coordinates]
    fused = embedded + other[:, :, :]
    output[:, :, :] = I.tanh(fused)


@intent.kernel
def fused_embedding_add_tanh_padded(
    input_indices: I.In[I.i64, ("M", "N")],
    weight: I.In[I.f32, ("V", "D")],
    other: I.In[I.f32, ("M", "N", "D")],
    padding_idx: I.i64,
    output: I.Out[I.f32, ("M", "N", "D")],
) -> None:
    rows = input_indices[:, :]
    row_coordinates = I.reshape(rows, (input_indices.shape[0], input_indices.shape[1], 1))
    embedding_coordinates = I.reshape(
        I.indices(I.domain(0, weight.shape[1])),
        (1, 1, weight.shape[1]),
    )
    embedded = weight[row_coordinates, embedding_coordinates]
    padding_mask = I.reshape(
        rows == padding_idx,
        (input_indices.shape[0], input_indices.shape[1], 1),
    )
    zero_embeddings = I.full(embedded.shape, 0.0, I.f32)
    embedded = I.select(padding_mask, zero_embeddings, embedded)
    fused = embedded + other[:, :, :]
    output[:, :, :] = I.tanh(fused)


def build(context):
    default_artifact = context.compile(
        "fused_embedding_add_tanh_default",
        fused_embedding_add_tanh_default,
    )
    padded_artifact = context.compile(
        "fused_embedding_add_tanh_padded",
        fused_embedding_add_tanh_padded,
    )

    def fused_embedding_add_tanh(
        input_indices,
        weight,
        other,
        *,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        out=None,
    ):
        if max_norm is not None:
            raise ValueError("max_norm is not supported by this fixed forward profile")

        if padding_idx is None:
            if out is None:
                return default_artifact.run(input_indices, weight, other)
            default_artifact(input_indices, weight, other, out)
            return out

        if out is None:
            return padded_artifact.run(
                input_indices,
                weight,
                other,
                padding_idx,
            )
        padded_artifact(input_indices, weight, other, padding_idx, out)
        return out

    return fused_embedding_add_tanh
