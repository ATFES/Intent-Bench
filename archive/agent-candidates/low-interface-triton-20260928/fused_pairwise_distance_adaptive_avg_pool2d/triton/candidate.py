import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    x1, x2, out,
    n, c, h, w, oh, ow,
    p, eps,
    keepdim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    total = oh * ow * c
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for base in tl.range(0, total, BLOCK):
        q = base + offs
        m = q < total
        spatial = q // c
        ci = q - spatial * c
        oy = spatial // ow
        ox = spatial - oy * ow

        hs = (oy * h) // oh
        he = ((oy + 1) * h + oh - 1) // oh
        ws = (ox * w) // ow
        we = ((ox + 1) * w + ow - 1) // ow

        a = tl.zeros([BLOCK], dtype=tl.float32)
        b = tl.zeros([BLOCK], dtype=tl.float32)
        for iy in tl.range(0, h):
            for ix in tl.range(0, w):
                inside = (iy >= hs) & (iy < he) & (ix >= ws) & (ix < we) & m
                off = ((pid * c + ci) * h + iy) * w + ix
                a += tl.load(x1 + off, mask=inside, other=0.0)
                b += tl.load(x2 + off, mask=inside, other=0.0)
        count = (he - hs) * (we - ws)
        d = (a / count - b / count) + eps
        v = tl.abs(d)
        acc += tl.exp(tl.log(v) * p)

    norm = tl.exp(tl.log(tl.sum(acc, axis=0)) / p)
    if keepdim:
        tl.store(out + pid, norm, mask=pid < n)
    else:
        tl.store(out + pid, norm, mask=pid < n)


def build(context):
    def fused_pairwise_distance_adaptive_avg_pool2d(
        x1: torch.Tensor,
        x2: torch.Tensor,
        output_size: int or tuple,
        p: float = 2.0,
        eps: float = 1e-6,
        keepdim: bool = False,
    ) -> torch.Tensor:
        if isinstance(output_size, int):
            oh = ow = output_size
        else:
            oh, ow = output_size
        n, c, h, w = x1.shape
        shape = (n, 1) if keepdim else (n,)
        out = torch.empty(shape, device=x1.device, dtype=x1.dtype)
        _kernel[(n,)](x1, x2, out, n, c, h, w, oh, ow, p, eps, keepdim=keepdim, BLOCK=1024)
        return out

    return fused_pairwise_distance_adaptive_avg_pool2d
