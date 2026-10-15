import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wm + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k_block * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & k_mask[:, None], other=0.0)
        acc = tl.dot(x, w, acc=acc, input_precision="ieee")
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    out_ptr,
    rows,
    cols,
    stride_x,
    stride_out,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    x = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / cols
    normalized = centered * tl.rsqrt(variance + eps)
    tl.store(out_ptr + row * stride_out + offsets, normalized, mask=mask)


def build(context):
    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        rows = input.shape[0]
        cols = weight.shape[0]
        intermediate = torch.empty((rows, cols), device=input.device, dtype=torch.float32)
        output = torch.empty((rows, cols), device=input.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_M']) * triton.cdiv(cols, meta['BLOCK_N']),)
        _linear_relu_kernel[grid](
            input,
            weight,
            bias,
            intermediate,
            rows,
            cols,
            input.shape[1],
            input.stride(0),
            weight.stride(0),
            intermediate.stride(0),
            GROUP_M=8,
        )
        _layer_norm_kernel[(rows,)](
            intermediate,
            output,
            rows,
            cols,
            intermediate.stride(0),
            output.stride(0),
            eps,
            BLOCK=2048,
            num_warps=8,
        )
        return output

    return wrapper
