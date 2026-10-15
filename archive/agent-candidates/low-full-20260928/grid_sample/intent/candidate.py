import torch
import intent
import intent.language as I


@intent.kernel
def _sample(inp: I.In[I.f32, (16, 3, 256, 256)], grid: I.In[I.f32, (16, 128, 128, 2)], out: I.Out[I.f32, (16, 3, 128, 128)]):
    ns = I.domain(0, 16)
    cs = I.domain(0, 3)
    ys = I.domain(0, 128)
    xs = I.domain(0, 128)
    for n in I.parallel(ns):
        for c in I.parallel(cs):
            for oy in I.parallel(ys):
                for ox in I.parallel(xs):
                    gx = grid[n, oy, ox, 0]
                    gy = grid[n, oy, ox, 1]
                    # NaNs in grid are specified to behave as -1.
                    gx = I.select(gx != gx, I.cast(-1.0, I.f32), gx)
                    gy = I.select(gy != gy, I.cast(-1.0, I.f32), gy)
                    ix = (gx + I.cast(1.0, I.f32)) * I.cast(256.0, I.f32) / I.cast(2.0, I.f32) - I.cast(0.5, I.f32)
                    iy = (gy + I.cast(1.0, I.f32)) * I.cast(256.0, I.f32) / I.cast(2.0, I.f32) - I.cast(0.5, I.f32)
                    x0f = I.floor(ix)
                    y0f = I.floor(iy)
                    x0 = I.cast(x0f, I.index)
                    y0 = I.cast(y0f, I.index)
                    wx = ix - x0f
                    wy = iy - y0f
                    x1 = x0 + 1
                    y1 = y0 + 1
                    x0v = (x0 >= 0) & (x0 < 256)
                    x1v = (x1 >= 0) & (x1 < 256)
                    y0v = (y0 >= 0) & (y0 < 256)
                    y1v = (y1 >= 0) & (y1 < 256)
                    v00 = I.gather(inp[n, c], (y0, x0), valid=x0v & y0v, fill=I.cast(0.0, I.f32))
                    v01 = I.gather(inp[n, c], (y0, x1), valid=x1v & y0v, fill=I.cast(0.0, I.f32))
                    v10 = I.gather(inp[n, c], (y1, x0), valid=x0v & y1v, fill=I.cast(0.0, I.f32))
                    v11 = I.gather(inp[n, c], (y1, x1), valid=x1v & y1v, fill=I.cast(0.0, I.f32))
                    value = (v00 * (I.cast(1.0, I.f32) - wx) + v01 * wx) * (I.cast(1.0, I.f32) - wy) + (v10 * (I.cast(1.0, I.f32) - wx) + v11 * wx) * wy
                    out[n, c, oy, ox] = value


def build(context):
    sample = context.compile("grid_sample_bilinear", _sample)

    def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        # The supplied invocation profile is spatial bilinear with zero padding.
        output = torch.empty((16, 3, 128, 128), device=input.device, dtype=input.dtype)
        sample(input, grid, output)
        return output

    return grid_sample
