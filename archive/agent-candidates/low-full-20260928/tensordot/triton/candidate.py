import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _tensordot_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                    mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
    out = acc.to(tl.float16)
    tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
             out, mask=(rows[:, None] < M) & (cols[None, :] < N))


def build(context):
    def wrapper(a, b, dims):
        # The supplied invocation contracts the trailing two axes of a with
        # the leading two axes of b, yielding a 1024 x 1024 matrix.
        if isinstance(dims, int):
            dim_count = dims
        elif torch.is_tensor(dims):
            dim_count = int(dims.item())
        else:
            left, right = dims
            if list(left) != [1, 2] or list(right) != [0, 1]:
                raise ValueError("unsupported dimensions for fixed profile")
            dim_count = len(left)
        if dim_count != 2 or a.ndim != 3 or b.ndim != 3:
            raise ValueError("unsupported dimensions for fixed profile")
        M, K = a.shape[0], a.shape[1] * a.shape[2]
        Kb, N = b.shape[0] * b.shape[1], b.shape[2]
        if K != Kb:
            raise ValueError("contracted dimensions must match")
        a2 = a.reshape(M, K)
        b2 = b.reshape(K, N)
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                             triton.cdiv(N, meta['BLOCK_N']))
        _tensordot_kernel[grid](
            a2, b2, out, M, N, K,
            a2.stride(0), a2.stride(1), b2.stride(0), b2.stride(1),
            out.stride(0), out.stride(1),
        )
        return out

    return wrapper
