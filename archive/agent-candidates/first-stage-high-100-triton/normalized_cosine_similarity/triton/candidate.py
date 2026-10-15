import torch
import triton
import triton.language as tl


@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    stride_x10,
    stride_x11,
    stride_x20,
    stride_x21,
    stride_out,
    n_cols,
    eps_similarity,
    eps_norm,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x1 = tl.load(x1_ptr + row * stride_x10 + cols * stride_x11, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + row * stride_x20 + cols * stride_x21, mask=mask, other=0.0)
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)

    norm1 = tl.sqrt(tl.sum(x1 * x1, axis=0))
    norm2 = tl.sqrt(tl.sum(x2 * x2, axis=0))
    raw_norm1 = norm1
    raw_norm2 = norm2
    norm1 = tl.maximum(norm1, eps_norm)
    norm2 = tl.maximum(norm2, eps_norm)
    x1 = x1 / norm1
    x2 = x2 / norm2

    dot = tl.sum(x1 * x2, axis=0)
    # For p_norm=2, normalization scales the original L2 norm by
    # 1 / max(original_norm, eps_norm).  Reusing the reductions above
    # preserves the zero-vector behavior while avoiding two extra reductions.
    normalized_norm1 = raw_norm1 / norm1
    normalized_norm2 = raw_norm2 / norm2
    normalized_norm1 = tl.maximum(normalized_norm1, eps_similarity)
    normalized_norm2 = tl.maximum(normalized_norm2, eps_similarity)
    result = dot / (normalized_norm1 * normalized_norm2)
    tl.store(out_ptr + row * stride_out, result)


def build(context):
    def wrapper(x1, x2, dim=1, eps_similarity=1e-8, p_norm=2, eps_norm=1e-12):
        output = torch.empty((x1.shape[0],), device=x1.device, dtype=x1.dtype)
        _normalized_cosine_similarity_kernel[(x1.shape[0],)](
            x1,
            x2,
            output,
            x1.stride(0),
            x1.stride(1),
            x2.stride(0),
            x2.stride(1),
            output.stride(0),
            x1.shape[1],
            eps_similarity,
            eps_norm,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
