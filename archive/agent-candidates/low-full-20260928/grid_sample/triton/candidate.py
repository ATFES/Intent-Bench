import torch
import triton
import triton.language as tl


@triton.jit
def _grid_sample_2d(
    inp, grid, out, n_elements,
    N, C, IH, IW, OH, OW,
    MODE: tl.constexpr, PAD: tl.constexpr, ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    c = offs % C
    q = offs // C
    ow = q % OW
    q = q // OW
    oh = q % OH
    n = q // OH

    gp = (n * OH + oh) * OW + ow
    gx = tl.load(grid + gp * 2, mask=mask, other=0.0)
    gy = tl.load(grid + gp * 2 + 1, mask=mask, other=0.0)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    if ALIGN:
        ix = (gx + 1.0) * (IW - 1) * 0.5
        iy = (gy + 1.0) * (IH - 1) * 0.5
    else:
        ix = ((gx + 1.0) * IW - 1.0) * 0.5
        iy = ((gy + 1.0) * IH - 1.0) * 0.5

    if MODE == 0:
        xi = tl.cast(tl.floor(ix + 0.5), tl.int32)
        yi = tl.cast(tl.floor(iy + 0.5), tl.int32)
        valid = (xi >= 0) & (xi < IW) & (yi >= 0) & (yi < IH)
        if PAD == 1:
            xi = tl.maximum(0, tl.minimum(IW - 1, xi))
            yi = tl.maximum(0, tl.minimum(IH - 1, yi))
            valid = mask
        else:
            valid = valid & mask
        addr = ((n * C + c) * IH + yi) * IW + xi
        value = tl.load(inp + addr, mask=valid, other=0.0)
    else:
        if PAD == 1:
            ix = tl.maximum(0.0, tl.minimum(float(IW - 1), ix))
            iy = tl.maximum(0.0, tl.minimum(float(IH - 1), iy))
        x0 = tl.cast(tl.floor(ix), tl.int32)
        y0 = tl.cast(tl.floor(iy), tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = ix - x0.to(tl.float32)
        wy = iy - y0.to(tl.float32)
        v00 = (x0 >= 0) & (x0 < IW) & (y0 >= 0) & (y0 < IH)
        v01 = (x1 >= 0) & (x1 < IW) & (y0 >= 0) & (y0 < IH)
        v10 = (x0 >= 0) & (x0 < IW) & (y1 >= 0) & (y1 < IH)
        v11 = (x1 >= 0) & (x1 < IW) & (y1 >= 0) & (y1 < IH)
        base = (n * C + c) * IH * IW
        a = tl.load(inp + base + y0 * IW + x0, mask=v00 & mask, other=0.0)
        b = tl.load(inp + base + y0 * IW + x1, mask=v01 & mask, other=0.0)
        d = tl.load(inp + base + y1 * IW + x0, mask=v10 & mask, other=0.0)
        e = tl.load(inp + base + y1 * IW + x1, mask=v11 & mask, other=0.0)
        value = a * (1.0 - wx) * (1.0 - wy) + b * wx * (1.0 - wy)
        value += d * (1.0 - wx) * wy + e * wx * wy

    tl.store(out + offs, value, mask=mask)


def build(context):
    def wrapper(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
        n, c, ih, iw = input.shape
        oh, ow = grid.shape[1], grid.shape[2]
        output = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
        mode_id = 0 if mode == 'nearest' else 1
        pad_id = 1 if padding_mode == 'border' else 0
        count = n * c * oh * ow
        _grid_sample_2d[(triton.cdiv(count, 128),)](
            input, grid, output, count, n, c, ih, iw, oh, ow,
            MODE=mode_id, PAD=pad_id, ALIGN=bool(align_corners), BLOCK=128,
        )
        return output
    return wrapper
