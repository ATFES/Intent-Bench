import torch
import intent
import intent.language as I


@intent.kernel
def _grid_sample_bilinear_zeros(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    grid: I.In[I.f32, ("N", "OH", "OW", 2)],
    output: I.Out[I.f32, ("N", "C", "OH", "OW")],
):
    N, C, H, W = input.shape
    OH = grid.shape[1]
    OW = grid.shape[2]

    h_index = I.cast(H, I.index)
    w_index = I.cast(W, I.index)
    h_float = I.cast(H, I.f32)
    w_float = I.cast(W, I.f32)
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)
    zero = I.cast(0.0, I.f32)
    minus_one = I.cast(-1.0, I.f32)

    for n in I.parallel(I.domain(0, N)):
        for oh in I.parallel(I.domain(0, OH)):
            for ow in I.parallel(I.domain(0, OW)):
                gx_raw = grid[n, oh, ow, 0]
                gy_raw = grid[n, oh, ow, 1]
                gx = I.select(gx_raw != gx_raw, minus_one, gx_raw)
                gy = I.select(gy_raw != gy_raw, minus_one, gy_raw)

                # align_corners=False: unnormalize to pixel-center coordinates.
                ix = (gx + one) * w_float * half - half
                iy = (gy + one) * h_float * half - half
                ix0 = I.cast(I.floor(ix), I.index)
                iy0 = I.cast(I.floor(iy), I.index)
                ix1 = ix0 + 1
                iy1 = iy0 + 1
                wx = ix - I.cast(ix0, I.f32)
                wy = iy - I.cast(iy0, I.f32)

                x0_valid = I.select(ix0 >= 0, ix0 < w_index, False)
                x1_valid = I.select(ix1 >= 0, ix1 < w_index, False)
                y0_valid = I.select(iy0 >= 0, iy0 < h_index, False)
                y1_valid = I.select(iy1 >= 0, iy1 < h_index, False)

                w00 = (one - wy) * (one - wx)
                w01 = (one - wy) * wx
                w10 = wy * (one - wx)
                w11 = wy * wx

                for c in I.domain(0, C):
                    v00 = I.gather(input, (n, c, iy0, ix0), valid=I.select(y0_valid, x0_valid, False), fill=zero)
                    v01 = I.gather(input, (n, c, iy0, ix1), valid=I.select(y0_valid, x1_valid, False), fill=zero)
                    v10 = I.gather(input, (n, c, iy1, ix0), valid=I.select(y1_valid, x0_valid, False), fill=zero)
                    v11 = I.gather(input, (n, c, iy1, ix1), valid=I.select(y1_valid, x1_valid, False), fill=zero)
                    value = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
                    output[n, c, oh, ow] = value


def build(context):
    artifact = context.compile("grid_sample_bilinear_zeros", _grid_sample_bilinear_zeros)

    def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            device=input.device,
            dtype=input.dtype,
        )
        artifact(input, grid, output)
        return output

    return grid_sample
