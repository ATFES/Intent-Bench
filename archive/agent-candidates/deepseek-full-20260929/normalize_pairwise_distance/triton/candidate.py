import torch
import triton
import triton.language as tl


@triton.jit
def pairwise_l2_kernel(
    x1_ptr,
    x2_ptr,
    dist_ptr,
    D: tl.constexpr,
    eps_distance,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    row = pid * D
    x1 = tl.load(x1_ptr + row + offs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + row + offs, mask=mask, other=0.0)

    diff = x1 - x2
    shifted = diff + eps_distance
    shifted = tl.where(mask, shifted, 0.0)

    sumsq = tl.sum(shifted * shifted, axis=0)
    distance = tl.sqrt_rn(sumsq)
    tl.store(dist_ptr + pid, distance)


@triton.jit
def normalize_l2_rows_kernel(
    dist_ptr,
    out_ptr,
    N: tl.constexpr,
    eps_norm,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    dist = tl.load(dist_ptr + b * N + offs, mask=mask, other=0.0)
    sumsq = tl.sum(dist * dist, axis=0)
    norm = tl.sqrt_rn(sumsq)
    denom = tl.maximum(norm, eps_norm)
    out = dist / denom
    tl.store(out_ptr + b * N + offs, out, mask=mask)


def normalize_pairwise_distance(
    x1,
    x2,
    p_distance=2.0,
    eps_distance=1e-06,
    keepdim=False,
    p_norm=2,
    dim_norm=1,
    eps_norm=1e-12,
):
    if p_distance != 2.0 or p_norm != 2:
        raise NotImplementedError(
            "This candidate implements the fixed p_distance=2, p_norm=2 profile"
        )

    if dim_norm != 1:
        raise NotImplementedError(
            "This candidate implements normalization along the fixed dim_norm=1 profile"
        )

    B, N, D = x1.shape

    dist = torch.empty((B, N), device=x1.device, dtype=torch.float32)
    out = torch.empty((B, N), device=x1.device, dtype=torch.float32)

    pairwise_l2_kernel[(B * N,)](
        x1_ptr=x1,
        x2_ptr=x2,
        dist_ptr=dist,
        D=D,
        eps_distance=float(eps_distance),
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=2,
    )

    normalize_l2_rows_kernel[(B,)](
        dist_ptr=dist,
        out_ptr=out,
        N=N,
        eps_norm=float(eps_norm),
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=8,
    )

    if keepdim:
        return out.unsqueeze(-1)
    return out


def build(context):
    return normalize_pairwise_distance
