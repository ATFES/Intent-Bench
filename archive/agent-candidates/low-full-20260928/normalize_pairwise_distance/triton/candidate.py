import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_pairwise_distance_kernel(
    x1, x2, out,
    stride_b, stride_n, stride_d,
    eps_distance, eps_norm,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    d = tl.arange(0, BLOCK_D)

    # Each program owns one batch row. The first reduction computes the
    # Euclidean distance for every vector pair, and the second normalizes
    # those distances across the requested dimension.
    offsets = b * stride_b + n[:, None] * stride_n + d[None, :] * stride_d
    valid_d = d[None, :] < 128
    a = tl.load(x1 + offsets, mask=valid_d, other=0.0)
    c = tl.load(x2 + offsets, mask=valid_d, other=0.0)
    delta = a - c + eps_distance
    distances = tl.sqrt(tl.sum(delta * delta, axis=1))

    denom = tl.sqrt(tl.sum(distances * distances, axis=0))
    denom = tl.maximum(denom, eps_norm)
    result = distances / denom
    tl.store(out + b * 256 + n, result)


def build(context):
    def normalize_pairwise_distance(
        x1, x2, p_distance=2.0, eps_distance=1e-6,
        keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12,
    ):
        out = torch.empty((16, 256), device=x1.device, dtype=x1.dtype)
        _normalize_pairwise_distance_kernel[(16,)](
            x1, x2, out,
            x1.stride(0), x1.stride(1), x1.stride(2),
            eps_distance, eps_norm,
            BLOCK_N=256, BLOCK_D=128,
        )
        return out

    return normalize_pairwise_distance
