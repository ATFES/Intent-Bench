from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.fn
def _unnormalize(
    value: I.f32,
    extent: I.f32,
    align_corners: I.Constexpr[bool],
) -> I.f32:
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)
    if align_corners:
        return (value + one) * (extent - one) * half
    return ((value + one) * extent - one) * half


@intent.kernel
def _grid_sample(
    input: I.In[I.f32, ("N", "C", "IN_H", "IN_W")],
    grid: I.In[I.f32, ("N", "OUT_H", "OUT_W", 2)],
    output: I.Out[I.f32, ("N", "C", "OUT_H", "OUT_W")],
    align_corners: I.Constexpr[bool],
) -> None:
    in_height = I.cast(input.shape[2], I.f32)
    in_width = I.cast(input.shape[3], I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)

    for batch in I.parallel(I.domain(0, input.shape[0])):
        for channel in I.parallel(I.domain(0, input.shape[1])):
            for out_y in I.parallel(I.domain(0, grid.shape[1])):
                for out_x in I.parallel(I.domain(0, grid.shape[2])):
                    grid_x = grid[batch, out_y, out_x, 0]
                    grid_y = grid[batch, out_y, out_x, 1]
                    grid_x = I.select(grid_x == grid_x, grid_x, -one)
                    grid_y = I.select(grid_y == grid_y, grid_y, -one)

                    source_x = _unnormalize(grid_x, in_width, align_corners)
                    source_y = _unnormalize(grid_y, in_height, align_corners)
                    floor_x = I.floor(source_x)
                    floor_y = I.floor(source_y)
                    x_valid = (floor_x >= zero) & (floor_x < in_width)
                    y_valid = (floor_y >= zero) & (floor_y < in_height)

                    x0 = I.cast(I.select(x_valid, floor_x, zero), I.index)
                    x1 = I.cast(
                        I.select(x_valid, floor_x + one, zero),
                        I.index,
                    )
                    y0 = I.cast(I.select(y_valid, floor_y, zero), I.index)
                    y1 = I.cast(
                        I.select(y_valid, floor_y + one, zero),
                        I.index,
                    )
                    x_weight_0 = I.select(x_valid, floor_x + one - source_x, zero)
                    x_weight_1 = I.select(x_valid, source_x - floor_x, zero)
                    y_weight_0 = I.select(y_valid, floor_y + one - source_y, zero)
                    y_weight_1 = I.select(y_valid, source_y - floor_y, zero)

                    value_00 = input[batch, channel, y0, x0]
                    value_01 = input[batch, channel, y0, x1]
                    value_10 = input[batch, channel, y1, x0]
                    value_11 = input[batch, channel, y1, x1]
                    output[batch, channel, out_y, out_x] = (
                        value_00 * y_weight_0 * x_weight_0
                        + value_01 * y_weight_0 * x_weight_1
                        + value_10 * y_weight_1 * x_weight_0
                        + value_11 * y_weight_1 * x_weight_1
                    )


def build(context):
    artifact = context.compile(
        "grid_sample_bilinear_zeros_align_false",
        _grid_sample,
        constexprs={"align_corners": False},
    )

    def grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ):
        if mode != "bilinear" or padding_mode != "zeros" or align_corners:
            raise ValueError("candidate is specialized to bilinear zeros align_corners=False")
        result = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        artifact(input, grid, result)
        return result

    return grid_sample
