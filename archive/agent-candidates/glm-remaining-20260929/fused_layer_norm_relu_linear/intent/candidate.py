from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _linear_relu_kernel(
    source: I.In[I.f32, ("BATCH", "REDUCTION")],
    matrix: I.In[I.f32, ("WIDTH", "REDUCTION")],
    linear_bias: I.In[I.f32, ("WIDTH",)],
    activated: I.Out[I.f32, ("BATCH", "WIDTH")],
    HAS_LINEAR_BIAS: I.Constexpr[bool],
):
    rows = I.domain(0, activated.shape[0])
    columns = I.domain(0, activated.shape[1])
    product = I.matmul(
        source,
        matrix,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    if HAS_LINEAR_BIAS:
        biased = product + linear_bias
    else:
        biased = product
    activated[rows, columns] = I.maximum(biased, I.cast(0.0, I.f32))


@intent.kernel
def _layer_norm_kernel(
    activated: I.In[I.f32, ("BATCH", "WIDTH")],
    normalized: I.Out[I.f32, ("BATCH", "WIDTH")],
    eps: I.f32,
    inv_width: I.f32,
    affine_scale: I.f32,
    affine_shift: I.f32,
    ELEMENTWISE_AFFINE: I.Constexpr[bool],
):
    rows = I.domain(0, activated.shape[0])
    columns = I.domain(0, activated.shape[1])
    values = activated[rows, columns]
    row_sum = I.reduce.sum(values, axis=1, acc_dtype=I.f32)
    mean = row_sum * inv_width
    centered = values - mean
    variance = I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32) * inv_width
    normalized_values = centered * I.rsqrt(variance + eps)
    if ELEMENTWISE_AFFINE:
        normalized[rows, columns] = (
            normalized_values * affine_scale + affine_shift
        )
    else:
        normalized[rows, columns] = normalized_values


def build(context):
    linear_with_bias = context.compile(
        "fused_linear_relu_with_bias",
        _linear_relu_kernel,
        constexprs={"HAS_LINEAR_BIAS": True},
    )
    linear_without_bias = context.compile(
        "fused_linear_relu_without_bias",
        _linear_relu_kernel,
        constexprs={"HAS_LINEAR_BIAS": False},
    )
    layer_norm_affine = context.compile(
        "fused_layer_norm_affine",
        _layer_norm_kernel,
        constexprs={"ELEMENTWISE_AFFINE": True},
    )
    layer_norm_identity = context.compile(
        "fused_layer_norm_identity",
        _layer_norm_kernel,
        constexprs={"ELEMENTWISE_AFFINE": False},
    )

    def fused_layer_norm_relu_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        normalized_shape: int | list[int] | torch.Size | None = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.Tensor:
        if input.ndim < 1 or weight.ndim != 2:
            raise ValueError("input must be (*, in_features) and weight must be 2-D")
        if input.shape[-1] != weight.shape[1]:
            raise ValueError("input and weight reduction dimensions do not match")

        batch = input.numel() // input.shape[-1]
        width = weight.shape[0]

        if normalized_shape is None:
            normalization_shape = torch.Size((width,))
        elif isinstance(normalized_shape, int):
            normalization_shape = torch.Size((normalized_shape,))
        else:
            normalization_shape = torch.Size(normalized_shape)
        if len(normalization_shape) == 0:
            raise ValueError("normalized_shape must be non-empty")
        normalization_width = 1
        for extent in normalization_shape:
            normalization_width *= extent
        if normalization_width != width:
            raise ValueError("normalized_shape does not match the linear output width")

        activated = torch.empty(
            (batch, width), dtype=torch.float32, device=input.device
        )
        output = torch.empty(
            (*input.shape[:-1], width),
            dtype=torch.float32,
            device=input.device,
        )

        if bias is None:
            unused_bias = torch.empty(
                width, dtype=torch.float32, device=input.device
            )
            linear_without_bias(input, weight, unused_bias, activated)
        else:
            linear_with_bias(input, weight, bias, activated)

        if elementwise_affine:
            layer_norm_affine(
                activated,
                output.view(batch, width),
                float(eps),
                1.0 / float(width),
                1.0,
                0.0,
            )
        else:
            layer_norm_identity(
                activated,
                output.view(batch, width),
                float(eps),
                1.0 / float(width),
                1.0,
                0.0,
            )
        return output

    return fused_layer_norm_relu_linear
