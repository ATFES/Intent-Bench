import torch

import intent
import intent.language as I


@intent.kernel
def _grid_sample_bilinear_zeros_align_false(
    inp: I.In[I.f32, ("N", "C", "H", "W")],
    grid: I.In[I.f32, ("N", "OH", "OW", 2)],
    out: I.Out[I.f32, ("N", "C", "OH", "OW")],
):
    N, C, H, W = inp.shape
    _, OH, OW, _ = grid.shape

    n_axis = I.domain(0, N)
    c_axis = I.domain(0, C)
    oh_axis = I.domain(0, OH)
    ow_axis = I.domain(0, OW)

    n_g = I.reshape(I.indices(n_axis), (N, 1, 1, 1))
    c_g = I.reshape(I.indices(c_axis), (1, C, 1, 1))
    oh_g = I.reshape(I.indices(oh_axis), (1, 1, OH, 1))
    ow_g = I.reshape(I.indices(ow_axis), (1, 1, 1, OW))

    gx = grid[n_g, oh_g, ow_g, 0]
    gy = grid[n_g, oh_g, ow_g, 1]

    one = I.cast(1.0, I.f32)
    two = I.cast(2.0, I.f32)
    zero = I.cast(0.0, I.f32)

    gx = I.select(gx != gx, I.cast(-1.0, I.f32), gx)
    gy = I.select(gy != gy, I.cast(-1.0, I.f32), gy)

    W_f = I.cast(W, I.f32)
    H_f = I.cast(H, I.f32)

    x = I.fdiv((gx + one) * W_f - one, two)
    y = I.fdiv((gy + one) * H_f - one, two)

    x0_f = I.floor(x)
    y0_f = I.floor(y)
    x1_f = x0_f + one
    y1_f = y0_f + one

    wx0 = x1_f - x
    wx1 = x - x0_f
    wy0 = y1_f - y
    wy1 = y - y0_f

    valid_x0 = I.select(x0_f >= zero, x0_f < W_f, False)
    valid_x1 = I.select(x1_f >= zero, x1_f < W_f, False)
    valid_y0 = I.select(y0_f >= zero, y0_f < H_f, False)
    valid_y1 = I.select(y1_f >= zero, y1_f < H_f, False)

    w_max = W_f - one
    h_max = H_f - one
    x0c_f = I.minimum(I.maximum(x0_f, zero), w_max)
    x1c_f = I.minimum(I.maximum(x1_f, zero), w_max)
    y0c_f = I.minimum(I.maximum(y0_f, zero), h_max)
    y1c_f = I.minimum(I.maximum(y1_f, zero), h_max)

    x0i = I.cast(x0c_f, I.index)
    x1i = I.cast(x1c_f, I.index)
    y0i = I.cast(y0c_f, I.index)
    y1i = I.cast(y1c_f, I.index)

    v00 = inp[n_g, c_g, y0i, x0i]
    v01 = inp[n_g, c_g, y0i, x1i]
    v10 = inp[n_g, c_g, y1i, x0i]
    v11 = inp[n_g, c_g, y1i, x1i]

    valid00 = I.select(valid_y0, valid_x0, False)
    valid01 = I.select(valid_y0, valid_x1, False)
    valid10 = I.select(valid_y1, valid_x0, False)
    valid11 = I.select(valid_y1, valid_x1, False)

    w00 = I.select(valid00, wx0 * wy0, zero)
    w01 = I.select(valid01, wx1 * wy0, zero)
    w10 = I.select(valid10, wx0 * wy1, zero)
    w11 = I.select(valid11, wx1 * wy1, zero)

    result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    out[n_axis, c_axis, oh_axis, ow_axis] = result


def build(context):
    kernel = context.compile(
        "grid_sample_bilinear_zeros_align_false",
        _grid_sample_bilinear_zeros_align_false,
    )

    def grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ):
        if mode != "bilinear" or padding_mode != "zeros" or align_corners:
            raise NotImplementedError(
                "this submission specializes the fixed invocation: "
                "bilinear/zeros/align_corners=False"
            )
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        kernel(input, grid, output)
        return output

    return grid_sample
