import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _combined_activation_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    sxm,
    sxk,
    sw1k,
    sw1n,
    sw2,
    sb,
    sym,
    syn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x_ptr + rows[:, None] * sxm + ks[None, :] * sxk,
            mask=row_mask[:, None] & (ks[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            w1_ptr + ks[:, None] * sw1k + cols[None, :] * sw1n,
            mask=(ks[:, None] < K) & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")

    # Keep the activation in fp32.  This form only relies on the basic
    # exponential primitive and is numerically stable over sigmoid's [0, 1]
    # output range.
    activated = 1.0 / (1.0 + tl.exp(-acc))
    activated = 2.0 / (1.0 + tl.exp(-2.0 * activated)) - 1.0
    scale = tl.load(w2_ptr + cols * sw2, mask=col_mask, other=0.0)
    shift = tl.load(bias_ptr + cols * sb, mask=col_mask, other=0.0)
    result = activated * scale[None, :] + shift[None, :]

    tl.store(
        y_ptr + rows[:, None] * sym + cols[None, :] * syn,
        result,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def build(context):
    def wrapper(input, weight1, weight2, bias, *, out=None):
        # The fixed profile is contiguous and two-dimensional, but flattening
        # leading dimensions keeps the same kernel valid for batched inputs.
        m = input.numel() // input.shape[-1]
        k = input.shape[-1]
        n = weight1.shape[-1]

        if out is None:
            output = torch.empty(
                input.shape[:-1] + (n,),
                device=input.device,
                dtype=input.dtype,
            )
        else:
            output = out

        x2 = input.reshape(m, k)
        y2 = output.reshape(m, n)
        w2 = weight2.reshape(-1)
        b2 = bias.reshape(-1)

        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )
        _combined_activation_kernel[grid](
            x2,
            weight1,
            w2,
            b2,
            y2,
            m,
            n,
            k,
            x2.stride(0),
            x2.stride(1),
            weight1.stride(0),
            weight1.stride(1),
            w2.stride(0),
            b2.stride(0),
            y2.stride(0),
            y2.stride(1),
        )
        return output

    return wrapper
