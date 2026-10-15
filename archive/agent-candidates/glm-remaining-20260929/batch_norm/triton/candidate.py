from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_norm_apply(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    eps,
    elements_per_channel,
    channels,
    BLOCK_SIZE: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
):
    block = tl.program_id(0)
    channel = tl.program_id(1)
    in_bounds = channel < channels
    running_mean = tl.load(running_mean_ptr + channel, mask=in_bounds, other=0.0)
    running_var = tl.load(running_var_ptr + channel, mask=in_bounds, other=0.0)
    normalized_scale = tl.rsqrt(running_var + eps)
    scale = normalized_scale
    shift = -running_mean * normalized_scale
    if HAS_AFFINE:
        weight = tl.load(weight_ptr + channel, mask=in_bounds, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=in_bounds, other=0.0)
        scale = scale * weight
        shift = shift * weight + bias

    offset_in_channel = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offset_in_channel < elements_per_channel
    input_offset = channel * elements_per_channel + offset_in_channel

    value = tl.load(input_ptr + input_offset, mask=valid, other=0.0).to(tl.float32)
    result = value * scale + shift
    tl.store(output_ptr + input_offset, result.to(tl.float16), mask=valid)


def build(context):
    def batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
    ):
        output = torch.empty_like(input)
        channels = input.shape[1]
        elements_per_channel = input.numel() // channels
        affine_weight = weight if weight is not None else running_mean
        affine_bias = bias if bias is not None else running_var

        has_affine = weight is not None and bias is not None
        if training:
            raise NotImplementedError("training batch normalization is not used by this profile")

        block_size = 4096
        blocks_per_channel = triton.cdiv(elements_per_channel, block_size)
        _batch_norm_apply[(blocks_per_channel, channels)](
            input,
            output,
            affine_weight,
            affine_bias,
            running_mean,
            running_var,
            eps,
            elements_per_channel,
            channels,
            BLOCK_SIZE=block_size,
            HAS_AFFINE=has_affine,
        )
        return output

    return batch_norm
