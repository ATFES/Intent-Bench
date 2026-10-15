from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_batch_norm_eval_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCKS_PER_PLANE: tl.constexpr,
):
    pid = tl.program_id(0)
    plane = pid // BLOCKS_PER_PLANE
    spatial = pid - plane * BLOCKS_PER_PLANE
    channel = plane % C

    offsets = plane * HW + spatial * BLOCK + tl.arange(0, BLOCK)

    mean = tl.load(mean_ptr + channel)
    var = tl.load(var_ptr + channel)
    invstd = tl.rsqrt(var + eps)

    scale = invstd
    if HAS_WEIGHT:
        scale = scale * tl.load(weight_ptr + channel)

    shift = -mean * scale
    if HAS_BIAS:
        shift = shift + tl.load(bias_ptr + channel)

    values = tl.load(x_ptr + offsets)
    normalized = values * scale + shift
    result = normalized * tl.sigmoid(normalized)
    tl.store(y_ptr + offsets, result)


def build(context):
    def silu_batch_norm(
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

        num_batches = input.shape[0]
        num_channels = input.shape[1]
        plane_size = input.shape[2] * input.shape[3]
        num_planes = num_batches * num_channels

        block = 1024
        blocks_per_plane = plane_size // block
        grid = (num_planes * blocks_per_plane,)

        _silu_batch_norm_eval_kernel[grid](
            input,
            output,
            running_mean,
            running_var,
            weight if weight is not None else input,
            bias if bias is not None else input,
            C=num_channels,
            HW=plane_size,
            eps=eps,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            BLOCK=block,
            BLOCKS_PER_PLANE=blocks_per_plane,
        )
        return output

    return silu_batch_norm
