import torch
import intent
import intent.language as I


@intent.kernel
def grid_sample_kernel(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    grid: I.In[I.f32, ("N", "H_out", "W_out", "2")],
    output: I.Out[I.f32, ("N", "C", "H_out", "W_out")],
    align_corners: I.Constexpr[bool],
):
    N, C, H, W = input.shape
    _, H_out, W_out, _ = grid.shape

    for n, ho, wo, c in I.parallel(
        I.domain(0, N),
        I.domain(0, H_out),
        I.domain(0, W_out),
        I.domain(0, C),
    ):
        raw_x = grid[n, ho, wo, 0]
        raw_y = grid[n, ho, wo, 1]

        x = I.select(raw_x != raw_x, I.cast(-1.0, I.f32), raw_x)
        y = I.select(raw_y != raw_y, I.cast(-1.0, I.f32), raw_y)

        if align_corners:
            u = (x + 1.0) * 0.5 * I.cast(W - 1, I.f32)
            v = (y + 1.0) * 0.5 * I.cast(H - 1, I.f32)
        else:
            u = (x + 1.0) * 0.5 * I.cast(W, I.f32) - 0.5
            v = (y + 1.0) * 0.5 * I.cast(H, I.f32) - 0.5

        i = I.cast(I.floor(u), I.i64)
        j = I.cast(I.floor(v), I.i64)
        i1 = i + 1
        j1 = j + 1

        w00 = (I.cast(i1, I.f32) - u) * (I.cast(j1, I.f32) - v)
        w01 = (I.cast(i1, I.f32) - u) * (v - I.cast(j, I.f32))
        w10 = (u - I.cast(i, I.f32)) * (I.cast(j1, I.f32) - v)
        w11 = (u - I.cast(i, I.f32)) * (v - I.cast(j, I.f32))

        i_clamped = I.maximum(I.minimum(i, W - 1), 0)
        i1_clamped = I.maximum(I.minimum(i1, W - 1), 0)
        j_clamped = I.maximum(I.minimum(j, H - 1), 0)
        j1_clamped = I.maximum(I.minimum(j1, H - 1), 0)

        i_ok = (i >= 0) & (i < W)
        i1_ok = (i1 >= 0) & (i1 < W)
        j_ok = (j >= 0) & (j < H)
        j1_ok = (j1 >= 0) & (j1 < H)

        v00_valid = i_ok & j_ok
        v01_valid = i1_ok & j_ok
        v10_valid = i_ok & j1_ok
        v11_valid = i1_ok & j1_ok

        v00 = I.select(
            v00_valid,
            I.assume_in_bounds(input[n, c, j_clamped, i_clamped], input, 2),
            I.cast(0.0, I.f32),
        )
        v01 = I.select(
            v01_valid,
            I.assume_in_bounds(input[n, c, j_clamped, i1_clamped], input, 2),
            I.cast(0.0, I.f32),
        )
        v10 = I.select(
            v10_valid,
            I.assume_in_bounds(input[n, c, j1_clamped, i_clamped], input, 2),
            I.cast(0.0, I.f32),
        )
        v11 = I.select(
            v11_valid,
            I.assume_in_bounds(input[n, c, j1_clamped, i1_clamped], input, 2),
            I.cast(0.0, I.f32),
        )

        output[n, c, ho, wo] = (
            w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        )


def build(context):
    artifact = context.compile("grid_sample", grid_sample_kernel)

    def wrapper(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        N, C, H, W = input.shape
        _, H_out, W_out, _ = grid.shape
        output = torch.empty(N, C, H_out, W_out, dtype=torch.float32, device="cuda")
        artifact(input, grid, output, align_corners)
        return output

    return wrapper

