import torch
import triton
import triton.language as tl


@triton.jit
def grid_sample_bilinear_zeros_4d(
    input_ptr,
    grid_ptr,
    output_ptr,
    batch_size,
    input_height,
    input_width,
    output_height,
    output_width,
    channels,
    input_stride_batch,
    input_stride_channel,
    input_stride_height,
    input_stride_width,
    grid_stride_batch,
    grid_stride_y,
    grid_stride_x,
    grid_stride_coordinate,
    BLOCK_PIXELS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    pixel_index = tl.program_id(0) * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
    channel_index = tl.arange(0, BLOCK_CHANNELS)
    pixel_count = batch_size * output_height * output_width
    pixel_mask = pixel_index < pixel_count
    channel_mask = channel_index < channels
    output_mask = pixel_mask[:, None] & channel_mask[None, :]

    output_pixel_stride = output_width * channels
    batch_index = pixel_index // output_pixel_stride
    pixel_in_batch = pixel_index % output_pixel_stride
    output_y = pixel_in_batch // output_width
    output_x = pixel_in_batch % output_width

    grid_offset = (
        batch_index * grid_stride_batch
        + output_y * grid_stride_y
        + output_x * grid_stride_x
    )
    grid_x = tl.load(
        grid_ptr + grid_offset + grid_stride_coordinate,
        mask=pixel_mask,
        other=0.0,
    )
    grid_y = tl.load(
        grid_ptr + grid_offset + grid_stride_coordinate + grid_stride_coordinate,
        mask=pixel_mask,
        other=0.0,
    )
    grid_x = tl.where(grid_x == grid_x, grid_x, -1.0)
    grid_y = tl.where(grid_y == grid_y, grid_y, -1.0)

    inside = (
        (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    source_x = (grid_x + 1.0) * input_width - 0.5
    source_y = (grid_y + 1.0) * input_height - 0.5
    source_x = tl.where(inside, source_x, -1.0)
    source_y = tl.where(inside, source_y, -1.0)

    x0 = tl.floor(source_x).to(tl.int32)
    y0 = tl.floor(source_y).to(tl.int32)
    fractional_x = source_x - x0.to(tl.float32)
    fractional_y = source_y - y0.to(tl.float32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0_valid = (x0 >= 0) & (x0 < input_width)
    x1_valid = (x1 >= 0) & (x1 < input_width)
    y0_valid = (y0 >= 0) & (y0 < input_height)
    y1_valid = (y1 >= 0) & (y1 < input_height)
    x0_mask = x0_valid[:, None] & output_mask
    x1_mask = x1_valid[:, None] & output_mask
    y0_mask = y0_valid[:, None] & output_mask
    y1_mask = y1_valid[:, None] & output_mask

    channel_offset = channel_index[None, :] * input_stride_channel
    x0_offset = x0[:, None] * input_stride_width + channel_offset
    x1_offset = x1[:, None] * input_stride_width + channel_offset
    y0_offset = y0[:, None] * input_stride_height
    y1_offset = y1[:, None] * input_stride_height
    batch_offset = batch_index[:, None] * input_stride_batch

    top_left = tl.load(
        input_ptr + batch_offset + y0_offset + x0_offset,
        mask=x0_mask,
        other=0.0,
    )
    top_right = tl.load(
        input_ptr + batch_offset + y0_offset + x1_offset,
        mask=x1_mask,
        other=0.0,
    )
    bottom_left = tl.load(
        input_ptr + batch_offset + y1_offset + x0_offset,
        mask=x0_mask,
        other=0.0,
    )
    bottom_right = tl.load(
        input_ptr + batch_offset + y1_offset + x1_offset,
        mask=x1_mask,
        other=0.0,
    )

    top = top_left * (1.0 - fractional_x[:, None]) + top_right * fractional_x[:, None]
    bottom = bottom_left * (1.0 - fractional_x[:, None]) + bottom_right * fractional_x[:, None]
    result = top * (1.0 - fractional_y[:, None]) + bottom * fractional_y[:, None]

    output_offset = pixel_index[:, None] * channels + channel_index[None, :]
    tl.store(output_ptr + output_offset, result, mask=output_mask)


def build(context):
    def grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ):
        if mode != "bilinear" or padding_mode != "zeros" or align_corners:
            raise ValueError("candidate specializes bilinear, zeros, align_corners=False")
        if input.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
            raise ValueError("expected 4-D input and [N, H, W, 2] grid")
        if input.shape[0] != grid.shape[0]:
            raise ValueError("input and grid batch sizes must match")

        batch_size, channels, input_height, input_width = input.shape
        output_height, output_width = grid.shape[1:3]
        output = torch.empty(
            (batch_size, channels, output_height, output_width),
            dtype=input.dtype,
            device=input.device,
        )
        pixel_count = batch_size * output_height * output_width
        grid_sample_bilinear_zeros_4d[(triton.cdiv(pixel_count, 128),)](
            input,
            grid,
            output,
            batch_size,
            input_height,
            input_width,
            output_height,
            output_width,
            channels,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            grid.stride(0),
            grid.stride(1),
            grid.stride(2),
            grid.stride(3),
            BLOCK_PIXELS=128,
            BLOCK_CHANNELS=4,
        )
        return output

    return grid_sample
