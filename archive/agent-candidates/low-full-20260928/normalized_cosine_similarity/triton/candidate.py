import torch
import triton
import triton.language as tl


@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    n_cols,
    eps_similarity,
    p_norm,
    eps_norm,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    base = row * n_cols + offsets

    x1 = tl.load(x1_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + base, mask=mask, other=0.0).to(tl.float32)

    # Compute both p-norms before normalizing, then perform the prescribed
    # cosine calculation on the normalized vectors.
    x1_abs_p = tl.math.pow(tl.abs(x1), p_norm)
    x2_abs_p = tl.math.pow(tl.abs(x2), p_norm)
    norm1 = tl.math.pow(tl.sum(x1_abs_p, axis=0), 1.0 / p_norm)
    norm2 = tl.math.pow(tl.sum(x2_abs_p, axis=0), 1.0 / p_norm)
    norm1 = tl.maximum(norm1, eps_norm)
    norm2 = tl.maximum(norm2, eps_norm)

    y1 = x1 / norm1
    y2 = x2 / norm2
    dot = tl.sum(y1 * y2, axis=0)
    l2_1 = tl.math.sqrt(tl.sum(y1 * y1, axis=0))
    l2_2 = tl.math.sqrt(tl.sum(y2 * y2, axis=0))
    denom = tl.maximum(l2_1, eps_similarity) * tl.maximum(l2_2, eps_similarity)
    tl.store(out_ptr + row, dot / denom)


def build(context):
    def wrapper(
        x1,
        x2,
        dim=1,
        eps_similarity=1e-8,
        p_norm=2,
        eps_norm=1e-12,
    ):
        # The supplied invocation is contiguous [rows, cols] and reduces dim 1.
        output = torch.empty((x1.shape[0],), device=x1.device, dtype=x1.dtype)
        _normalized_cosine_similarity_kernel[(x1.shape[0],)](
            x1,
            x2,
            output,
            x1.shape[1],
            eps_similarity,
            p_norm,
            eps_norm,
            BLOCK=1024,
        )
        return output

    return wrapper
