import torch
import triton
import triton.language as tl


@triton.jit
def _addmm_kernel(
    input_ptr,
    mat1_ptr,
    mat2_ptr,
    output_ptr,
    m,
    n,
    k,
    input_stride_m,
    input_stride_n,
    mat1_stride_m,
    mat1_stride_k,
    mat2_stride_k,
    mat2_stride_n,
    output_stride_m,
    output_stride_n,
    alpha,
    beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_MNK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

    input_offsets = offs_m[:, None] * input_stride_m + offs_n[None, :] * input_stride_n
    output_offsets = offs_m[:, None] * output_stride_m + offs_n[None, :] * output_stride_n
    if EVEN_MNK:
        input_mask = None
        output_mask = None
    else:
        input_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        output_mask = input_mask

    use_input = beta != 0.0
    if EVEN_MNK:
        input_values = tl.load(input_ptr + input_offsets, mask=None, other=0.0)
    else:
        input_values = tl.load(
            input_ptr + input_offsets,
            mask=input_mask & use_input,
            other=0.0,
        )

    mat1_offsets = mat1_ptr + offs_m[:, None] * mat1_stride_m + offs_k[None, :] * mat1_stride_k
    mat2_offsets = mat2_ptr + offs_k[:, None] * mat2_stride_k + offs_n[None, :] * mat2_stride_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k, BLOCK_K):
        if EVEN_MNK:
            mat1_values = tl.load(mat1_offsets)
            mat2_values = tl.load(mat2_offsets)
        else:
            mat1_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
            mat2_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)
            mat1_values = tl.load(mat1_offsets, mask=mat1_mask, other=0.0)
            mat2_values = tl.load(mat2_offsets, mask=mat2_mask, other=0.0)
        accumulator = tl.dot(mat1_values, mat2_values, accumulator, out_dtype=tl.float32)
        mat1_offsets += BLOCK_K * mat1_stride_k
        mat2_offsets += BLOCK_K * mat2_stride_k

    product = alpha * accumulator
    result = tl.where(use_input, beta * input_values.to(tl.float32) + product, product)
    if EVEN_MNK:
        tl.store(output_ptr + output_offsets, result.to(output_ptr.dtype.element_ty))
    else:
        tl.store(
            output_ptr + output_offsets,
            result.to(output_ptr.dtype.element_ty),
            mask=output_mask,
        )


def _broadcast_input_strides(tensor, m, n):
    if tensor.dim() == 0:
        return 0, 0
    if tensor.dim() == 1:
        if tensor.shape[0] == m:
            return tensor.stride(0), 0
        if tensor.shape[0] == n:
            return 0, tensor.stride(0)
        raise ValueError(f"input shape {tuple(tensor.shape)} is not broadcastable to ({m}, {n})")
    if tensor.dim() == 2:
        if tensor.shape[0] in (1, m) and tensor.shape[1] in (1, n):
            return tensor.stride(0), tensor.stride(1)
        raise ValueError(f"input shape {tuple(tensor.shape)} is not broadcastable to ({m}, {n})")
    raise ValueError("input must be at most two-dimensional")


def _launch_config(m, n, k):
    if m >= 128 and n >= 128 and k >= 64:
        return 128, 64, 64, 4, 4
    return 64, 64, 32, 4, 3


def build(context):
    def addmm(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
        if mat1.dim() != 2 or mat2.dim() != 2:
            raise ValueError("mat1 and mat2 must be two-dimensional")
        m, k = mat1.shape
        k2, n = mat2.shape
        if k != k2:
            raise ValueError("mat1 and mat2 have incompatible reduction dimensions")
        if input.dim() > 2:
            raise ValueError("input must be at most two-dimensional")

        output = out
        if output is None:
            output = torch.empty((m, n), device=mat1.device, dtype=mat1.dtype)
        elif output.shape != (m, n) or output.dtype != mat1.dtype or output.device != mat1.device:
            raise ValueError("out must have the matrix result shape, dtype, and device")

        block_m, block_n, block_k, num_warps, num_stages = _launch_config(m, n, k)
        _addmm_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
            input,
            mat1,
            mat2,
            output,
            m,
            n,
            k,
            *_broadcast_input_strides(input, m, n),
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            output.stride(0),
            output.stride(1),
            alpha,
            beta,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_MNK=(m % block_m == 0 and n % block_n == 0 and k % block_k == 0),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output

    return addmm
