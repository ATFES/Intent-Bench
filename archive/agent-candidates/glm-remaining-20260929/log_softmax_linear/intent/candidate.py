from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def _fused_with_bias(
    source: I.In[I.f32, ("BATCH", "IN_FEATURES")],
    matrix: I.In[I.f32, ("IN_FEATURES", "OUT_FEATURES")],
    offset: I.In[I.f32, ("OUT_FEATURES",)],
    destination: I.Out[I.f32, ("BATCH", "OUT_FEATURES")],
) -> None:
    logits = I.matmul(
        source,
        matrix,
        acc_dtype=I.f32,
        transpose_rhs=True,
    ) + offset

    row_max = I.reduce.max(logits, axis=1, acc_dtype=I.f32)
    row_max = I.reshape(row_max, (logits.shape[0], 1))
    stabilized = logits - row_max
    probabilities = I.exp(stabilized)
    normalizer = I.reduce.sum(probabilities, axis=1, acc_dtype=I.f32)
    normalizer = I.reshape(normalizer, (logits.shape[0], 1))
    destination[:, :] = stabilized - I.log(normalizer)


@intent.kernel
def _fused_without_bias(
    source: I.In[I.f32, ("BATCH", "IN_FEATURES")],
    matrix: I.In[I.f32, ("IN_FEATURES", "OUT_FEATURES")],
    destination: I.Out[I.f32, ("BATCH", "OUT_FEATURES")],
) -> None:
    logits = I.matmul(
        source,
        matrix,
        acc_dtype=I.f32,
        transpose_rhs=True,
    )

    row_max = I.reduce.max(logits, axis=1, acc_dtype=I.f32)
    row_max = I.reshape(row_max, (logits.shape[0], 1))
    stabilized = logits - row_max
    probabilities = I.exp(stabilized)
    normalizer = I.reduce.sum(probabilities, axis=1, acc_dtype=I.f32)
    normalizer = I.reshape(normalizer, (logits.shape[0], 1))
    destination[:, :] = stabilized - I.log(normalizer)


def build(context):
    with_bias = context.compile(
        "log_softmax_linear_fused_bias",
        _fused_with_bias,
    )
    without_bias = context.compile(
        "log_softmax_linear_fused_no_bias",
        _fused_without_bias,
    )

    def log_softmax_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        dim: int = -1,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if dim != -1:
            raise NotImplementedError("only dim=-1 is supported")
        if dtype is not None:
            raise NotImplementedError("only dtype=None is supported")

        batch_size, in_features = input.shape
        out_features, weight_in_features = weight.shape
        output = torch.empty(
            (batch_size, out_features),
            device=input.device,
            dtype=torch.float32,
        )
        if bias is None:
            without_bias(input, weight, output)
        else:
            with_bias(input, weight, bias, output)
        return output

    return log_softmax_linear
