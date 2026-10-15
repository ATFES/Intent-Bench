from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_add_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    other_ptr,
    output_ptr,
    alpha,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    spatial_size: tl.constexpr = H * W
    tile_count = tl.cdiv(spatial_size, BLOCK_N)
    program_id = tl.program_id(0)
    batch = program_id // tile_count
    spatial_tile = program_id % tile_count

    output_offsets = spatial_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < spatial_size
    output_y = output_offsets // W
    output_x = output_offsets % W
    channels = tl.arange(0, C)
    output_channels = tl.arange(0, OUTPUT_CHANNELS)
    weight_tile = tl.load(
        weight_ptr + channels[:, None] * (3 * 3) + output_channels[None, :]
    )
    accumulator = tl.zeros((BLOCK_N, OUTPUT_CHANNELS), dtype=tl.float32)

    for kernel_index in tl.static_range(9):
        kernel_y = kernel_index // 3
        kernel_x = kernel_index % 3
        input_y = output_y + kernel_y - 1
        input_x = output_x + kernel_x - 1
        valid = (
            (input_y >= 0)
            & (input_y < H)
            & (input_x >= 0)
            & (input_x < W)
            & output_mask
        )
        source_offsets = input_y * W + input_x
        patch = tl.load(
            input_ptr
            + batch * (C * spatial_size)
            + channels[:, None] * spatial_size
            + source_offsets[None, :],
            mask=valid[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            tl.trans(patch),
            weight_tile,
            acc=accumulator,
            input_precision="ieee",
        )

    result = tl.trans(accumulator)
    if HAS_BIAS:
        result += tl.load(bias_ptr + output_channels)
    other = tl.load(
        other_ptr
        + batch * (OUTPUT_CHANNELS * spatial_size)
        + output_channels[None, :] * spatial_size
        + output_offsets[:, None]
    )
    result += alpha * other

    tl.store(
        output_ptr
        + batch * (OUTPUT_CHANNELS * spatial_size)
        + output_channels[None, :] * spatial_size
        + output_offsets[:, None],
        result,
        mask=output_mask[:, None],
    )


def build(context):
    def conv2d_add(
        input,
        weight,
        bias=None,
        other=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        alpha=1,
        out=None,
    ):
        if tuple(input.shape) != (16, 32, 32, 32):
            raise ValueError("input does not match the supplied invocation profile")
        if tuple(weight.shape) != (32, 32, 3, 3):
            raise ValueError("weight does not match the supplied invocation profile")
        if not input.is_contiguous() or not weight.is_contiguous():
            raise ValueError("input and weight must be contiguous")
        if input.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError("input and weight must be float32")
        if stride != 1 or padding != 1 or dilation != 1 or groups != 1:
            raise ValueError("parameters do not match the supplied invocation profile")
        if bias is not None and (tuple(bias.shape) != (32,) or not bias.is_contiguous()):
            raise ValueError("bias does not match the supplied invocation profile")
        if not torch.is_tensor(other) or tuple(other.shape) != (16, 32, 32, 32):
            raise ValueError("other does not match the supplied invocation profile")
        if not other.is_contiguous() or other.dtype != torch.float32:
            raise ValueError("other must be contiguous float32")

        if out is None:
            output = torch.empty(
                (16, 32, 32, 32), dtype=torch.float32, device=input.device
            )
        else:
            if tuple(out.shape) != (16, 32, 32, 32) or out.dtype != torch.float32:
                raise ValueError("out does not match the supplied invocation profile")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            output = out

        _conv2d_add_kernel[(16 * triton.cdiv(1024, 64),)](
            input,
            weight,
            input if bias is None else bias,
            other,
            output,
            float(alpha),
            N=16,
            C=32,
            H=32,
            W=32,
            OUTPUT_CHANNELS=32,
            BLOCK_N=64,
            HAS_BIAS=bias is not None,
            num_warps=8,
        )
        return output

    return conv2d_add
