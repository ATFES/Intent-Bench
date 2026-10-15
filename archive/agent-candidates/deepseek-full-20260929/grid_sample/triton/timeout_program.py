from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as _libdevice
import torch


_MODE_BILINEAR = 0
_MODE_NEAREST = 1

_PAD_ZEROS = 0
_PAD_BORDER = 1
_PAD_REFLECTION = 2


@triton.jit
def _reflect_coord(coord, twice_low: tl.constexpr, twice_high: tl.constexpr):
    # Mirrors a continuous coordinate into the inclusive interval
    # [twice_low / 2, twice_high / 2].  The two bounds are passed doubled so
    # half-integer endpoints stay exact in the constant domain.
    coord_min = twice_low * 0.5
    coord_span = (twice_high - twice_low) * 0.5
    c = tl.abs(coord - coord_min)
    extra = _libdevice.fmod(c, coord_span)
    flips = tl.floor(c / coord_span).to(tl.int32)
    even = (flips & 1) == 0
    return tl.where(even, extra + coord_min, coord_span + coord_min - extra)


@triton.jit
def _compute_coord(coord, size: tl.constexpr, PADDING: tl.constexpr,
                   ALIGN_CORNERS: tl.constexpr):
    if PADDING == 0:
        return coord
    elif PADDING == 1:
        return tl.clamp(coord, 0.0, (size - 1) * 1.0)
    else:
        if size == 1:
            return coord * 0.0
        if ALIGN_CORNERS:
            low = 0
            high = 2 * (size - 1)
        else:
            low = -1
            high = 2 * size - 1
        r = _reflect_coord(coord, low, high)
        return tl.clamp(r, 0.0, (size - 1) * 1.0)


@triton.jit
def _safe_round(coord):
    bounded = tl.minimum(tl.maximum(coord, -2147483647.0), 2147483647.0)
    return _libdevice.rint(bounded).to(tl.int32)


@triton.jit
def _safe_floor(coord):
    # Keep the float-to-int cast portable even for grid coordinates far outside
    # the normalized range.  Such lanes are subsequently masked off by the
    # boundary predicate, so clamping the unused index is numerically harmless.
    bounded = tl.minimum(tl.maximum(coord, -2147483647.0), 2147483647.0)
    return tl.floor(bounded).to(tl.int32)


@triton.jit
def _grid_sample_2d_kernel(input_ptr, grid_ptr, output_ptr,
                           N: tl.constexpr, C: tl.constexpr,
                           H: tl.constexpr, W: tl.constexpr,
                           HO: tl.constexpr, WO: tl.constexpr,
                           MODE: tl.constexpr, PADDING: tl.constexpr,
                           ALIGN_CORNERS: tl.constexpr, BP: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BP + tl.arange(0, BP)
    total = N * HO * WO
    pmask = p < total

    ow = p % WO
    t = p // WO
    oh = t % HO
    n = t // HO

    grid_off = n * (HO * WO * 2) + oh * (WO * 2) + ow * 2
    gx = tl.load(grid_ptr + grid_off, mask=pmask, other=0.0)
    gy = tl.load(grid_ptr + grid_off + 1, mask=pmask, other=0.0)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    if ALIGN_CORNERS:
        x = gx * (W * 0.5 - 0.5) + (W * 0.5 - 0.5)
        y = gy * (H * 0.5 - 0.5) + (H * 0.5 - 0.5)
    else:
        x = gx * (W * 0.5) + (W * 0.5 - 0.5)
        y = gy * (H * 0.5) + (H * 0.5 - 0.5)

    xs = _compute_coord(x, W, PADDING, ALIGN_CORNERS)
    ys = _compute_coord(y, H, PADDING, ALIGN_CORNERS)

    if MODE == 0:
        xb = _safe_floor(xs)
        x0 = xb
        x1 = x0 + 1
        wx = xs - xb

        yb = _safe_floor(ys)
        y0 = yb
        y1 = y0 + 1
        wy = ys - yb

        mx0 = (x0 >= 0) & (x0 < W)
        mx1 = (x1 >= 0) & (x1 < W)
        my0 = (y0 >= 0) & (y0 < H)
        my1 = (y1 >= 0) & (y1 < H)

        w00 = (1.0 - wx) * (1.0 - wy)
        w01 = wx * (1.0 - wy)
        w10 = (1.0 - wx) * wy
        w11 = wx * wy

        for cc in tl.static_range(0, C):
            out_off = ((n * C + cc) * HO + oh) * WO + ow
            in_base = (n * C + cc) * H * W
            v00 = tl.load(input_ptr + in_base + y0 * W + x0,
                          mask=pmask & my0 & mx0, other=0.0)
            v01 = tl.load(input_ptr + in_base + y0 * W + x1,
                          mask=pmask & my0 & mx1, other=0.0)
            v10 = tl.load(input_ptr + in_base + y1 * W + x0,
                          mask=pmask & my1 & mx0, other=0.0)
            v11 = tl.load(input_ptr + in_base + y1 * W + x1,
                          mask=pmask & my1 & mx1, other=0.0)
            result = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
            tl.store(output_ptr + out_off, result, mask=pmask)
    else:
        ix = _safe_round(xs)
        iy = _safe_round(ys)
        valid = pmask & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        for cc in tl.static_range(0, C):
            out_off = ((n * C + cc) * HO + oh) * WO + ow
            in_off = (n * C + cc) * H * W + iy * W + ix
            v = tl.load(input_ptr + in_off, mask=valid, other=0.0)
            tl.store(output_ptr + out_off, v, mask=pmask)


@triton.jit
def _grid_sample_3d_kernel(input_ptr, grid_ptr, output_ptr,
                           N: tl.constexpr, C: tl.constexpr,
                           D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                           DO: tl.constexpr, HO: tl.constexpr, WO: tl.constexpr,
                           MODE: tl.constexpr, PADDING: tl.constexpr,
                           ALIGN_CORNERS: tl.constexpr, BP: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BP + tl.arange(0, BP)
    total = N * DO * HO * WO
    pmask = p < total

    ow = p % WO
    t = p // WO
    oh = t % HO
    t = t // HO
    od = t % DO
    n = t // DO

    grid_off = (n * (DO * HO * WO * 3) + od * (HO * WO * 3) +
                oh * (WO * 3) + ow * 3)
    gx = tl.load(grid_ptr + grid_off, mask=pmask, other=0.0)
    gy = tl.load(grid_ptr + grid_off + 1, mask=pmask, other=0.0)
    gz = tl.load(grid_ptr + grid_off + 2, mask=pmask, other=0.0)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)
    gz = tl.where(gz != gz, -1.0, gz)

    if ALIGN_CORNERS:
        x = gx * (W * 0.5 - 0.5) + (W * 0.5 - 0.5)
        y = gy * (H * 0.5 - 0.5) + (H * 0.5 - 0.5)
        z = gz * (D * 0.5 - 0.5) + (D * 0.5 - 0.5)
    else:
        x = gx * (W * 0.5) + (W * 0.5 - 0.5)
        y = gy * (H * 0.5) + (H * 0.5 - 0.5)
        z = gz * (D * 0.5) + (D * 0.5 - 0.5)

    xs = _compute_coord(x, W, PADDING, ALIGN_CORNERS)
    ys = _compute_coord(y, H, PADDING, ALIGN_CORNERS)
    zs = _compute_coord(z, D, PADDING, ALIGN_CORNERS)

    if MODE == 0:
        xb = _safe_floor(xs)
        x0 = xb
        x1 = x0 + 1
        wx = xs - xb

        yb = _safe_floor(ys)
        y0 = yb
        y1 = y0 + 1
        wy = ys - yb

        zb = _safe_floor(zs)
        z0 = zb
        z1 = z0 + 1
        wz = zs - zb

        mx0 = (x0 >= 0) & (x0 < W)
        mx1 = (x1 >= 0) & (x1 < W)
        my0 = (y0 >= 0) & (y0 < H)
        my1 = (y1 >= 0) & (y1 < H)
        mz0 = (z0 >= 0) & (z0 < D)
        mz1 = (z1 >= 0) & (z1 < D)

        w000 = (1.0 - wz) * (1.0 - wy) * (1.0 - wx)
        w001 = (1.0 - wz) * (1.0 - wy) * wx
        w010 = (1.0 - wz) * wy * (1.0 - wx)
        w011 = (1.0 - wz) * wy * wx
        w100 = wz * (1.0 - wy) * (1.0 - wx)
        w101 = wz * (1.0 - wy) * wx
        w110 = wz * wy * (1.0 - wx)
        w111 = wz * wy * wx

        for cc in tl.static_range(0, C):
            out_off = (((n * C + cc) * DO + od) * HO + oh) * WO + ow
            in_base = (n * C + cc) * D * H * W
            v000 = tl.load(input_ptr + in_base + z0 * H * W + y0 * W + x0,
                           mask=pmask & mz0 & my0 & mx0, other=0.0)
            v001 = tl.load(input_ptr + in_base + z0 * H * W + y0 * W + x1,
                           mask=pmask & mz0 & my0 & mx1, other=0.0)
            v010 = tl.load(input_ptr + in_base + z0 * H * W + y1 * W + x0,
                           mask=pmask & mz0 & my1 & mx0, other=0.0)
            v011 = tl.load(input_ptr + in_base + z0 * H * W + y1 * W + x1,
                           mask=pmask & mz0 & my1 & mx1, other=0.0)
            v100 = tl.load(input_ptr + in_base + z1 * H * W + y0 * W + x0,
                           mask=pmask & mz1 & my0 & mx0, other=0.0)
            v101 = tl.load(input_ptr + in_base + z1 * H * W + y0 * W + x1,
                           mask=pmask & mz1 & my0 & mx1, other=0.0)
            v110 = tl.load(input_ptr + in_base + z1 * H * W + y1 * W + x0,
                           mask=pmask & mz1 & my1 & mx0, other=0.0)
            v111 = tl.load(input_ptr + in_base + z1 * H * W + y1 * W + x1,
                           mask=pmask & mz1 & my1 & mx1, other=0.0)
            result = (v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011 +
                      v100 * w100 + v101 * w101 + v110 * w110 + v111 * w111)
            tl.store(output_ptr + out_off, result, mask=pmask)
    else:
        ix = _safe_round(xs)
        iy = _safe_round(ys)
        iz = _safe_round(zs)
        valid = (pmask & (ix >= 0) & (ix < W) &
                 (iy >= 0) & (iy < H) & (iz >= 0) & (iz < D))
        for cc in tl.static_range(0, C):
            out_off = (((n * C + cc) * DO + od) * HO + oh) * WO + ow
            in_off = ((n * C + cc) * D * H * W + iz * H * W + iy * W + ix)
            v = tl.load(input_ptr + in_off, mask=valid, other=0.0)
            tl.store(output_ptr + out_off, v, mask=pmask)


def _mode_code(mode):
    if mode == 'bilinear':
        return _MODE_BILINEAR
    if mode == 'nearest':
        return _MODE_NEAREST
    raise ValueError("grid_sample: expected mode to be 'bilinear' or 'nearest', "
                     f"but got {mode!r}")


def _padding_code(padding_mode):
    if padding_mode == 'zeros':
        return _PAD_ZEROS
    if padding_mode == 'border':
        return _PAD_BORDER
    if padding_mode == 'reflection':
        return _PAD_REFLECTION
    raise ValueError("grid_sample: expected padding_mode to be 'zeros', "
                     f"'border' or 'reflection', but got {padding_mode!r}")


def build(context):
    def grid_sample(input, grid, mode='bilinear', padding_mode='zeros',
                    align_corners=False):
        if not (input.is_cuda and grid.is_cuda):
            raise RuntimeError("grid_sample Triton implementation requires CUDA tensors")
        if not input.is_contiguous() or not grid.is_contiguous():
            raise RuntimeError("grid_sample Triton implementation requires contiguous tensors")

        mode_i = _mode_code(mode)
        padding_i = _padding_code(padding_mode)
        bp = 128
        num_warps = 4

        if input.dim() == 4:
            if grid.dim() != 4 or grid.shape[-1] != 2:
                raise ValueError("grid_sample: expected a 4D grid with last dimension 2")
            N, C, H, W = input.shape
            GO, HO, WO, _ = grid.shape
            if GO != N:
                raise ValueError("grid_sample: input and grid batch sizes differ")
            output = torch.empty((N, C, HO, WO), device=input.device,
                                 dtype=input.dtype)
            total = N * HO * WO
            _grid_sample_2d_kernel[(triton.cdiv(total, bp),)](
                input, grid, output,
                N=N, C=C, H=H, W=W, HO=HO, WO=WO,
                MODE=mode_i, PADDING=padding_i,
                ALIGN_CORNERS=align_corners, BP=bp,
                num_warps=num_warps)
            return output

        if input.dim() == 5:
            if grid.dim() != 5 or grid.shape[-1] != 3:
                raise ValueError("grid_sample: expected a 5D grid with last dimension 3")
            N, C, D, H, W = input.shape
            GO, DO, HO, WO, _ = grid.shape
            if GO != N:
                raise ValueError("grid_sample: input and grid batch sizes differ")
            output = torch.empty((N, C, DO, HO, WO), device=input.device,
                                 dtype=input.dtype)
            total = N * DO * HO * WO
            _grid_sample_3d_kernel[(triton.cdiv(total, bp),)](
                input, grid, output,
                N=N, C=C, D=D, H=H, W=W, DO=DO, HO=HO, WO=WO,
                MODE=mode_i, PADDING=padding_i,
                ALIGN_CORNERS=align_corners, BP=bp,
                num_warps=num_warps)
            return output

        raise ValueError("grid_sample: input must be a 4D or 5D tensor")

    return grid_sample
