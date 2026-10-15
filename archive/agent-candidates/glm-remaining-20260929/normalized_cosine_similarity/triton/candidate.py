from __future__ import annotations

import triton
import triton.language as tl
import torch

libdevice = tl.extra.cuda.libdevice

@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    columns,
    x1_row_stride,
    x1_col_stride,
    x2_row_stride,
    x2_col_stride,
    eps_similarity,
    eps_norm,
    P_NORM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    x1_row_ptr = x1_ptr + row * x1_row_stride
    x2_row_ptr = x2_ptr + row * x2_row_stride

    if P_NORM == 2.0:
        dot_sum = 0.0
        x1_square_sum = 0.0
        x2_square_sum = 0.0
        for block_start in range(0, columns, BLOCK_SIZE):
            offsets = block_start + column_offsets
            mask = offsets < columns
            x1 = tl.load(x1_row_ptr + offsets * x1_col_stride, mask=mask, other=0.0)
            x2 = tl.load(x2_row_ptr + offsets * x2_col_stride, mask=mask, other=0.0)
            dot_sum += tl.sum(x1 * x2)
            x1_square_sum += tl.sum(x1 * x1)
            x2_square_sum += tl.sum(x2 * x2)

        x1_norm = tl.sqrt_rn(x1_square_sum)
        x2_norm = tl.sqrt_rn(x2_square_sum)
        x1_scale = tl.maximum(x1_norm, eps_norm)
        x2_scale = tl.maximum(x2_norm, eps_norm)
        normalized_x1_norm = x1_norm / x1_scale
        normalized_x2_norm = x2_norm / x2_scale
        similarity_denominator = tl.maximum(normalized_x1_norm, eps_similarity) * tl.maximum(
            normalized_x2_norm, eps_similarity
        )
        result = (dot_sum / (x1_scale * x2_scale)) / similarity_denominator
        tl.store(out_ptr + row, result.to(out_ptr.dtype.element_ty))
    else:
        x1_p_norm_sum = 0.0
        x2_p_norm_sum = 0.0
        for block_start in range(0, columns, BLOCK_SIZE):
            offsets = block_start + column_offsets
            mask = offsets < columns
            x1 = tl.load(x1_row_ptr + offsets * x1_col_stride, mask=mask, other=0.0)
            x2 = tl.load(x2_row_ptr + offsets * x2_col_stride, mask=mask, other=0.0)
            x1_p_norm_sum += tl.sum(tl.abs(x1) ** P_NORM)
            x2_p_norm_sum += tl.sum(tl.abs(x2) ** P_NORM)

        x1_scale = tl.maximum(libdevice.pow(x1_p_norm_sum, 1.0 / P_NORM), eps_norm)
        x2_scale = tl.maximum(libdevice.pow(x2_p_norm_sum, 1.0 / P_NORM), eps_norm)

        dot_sum = 0.0
        x1_square_sum = 0.0
        x2_square_sum = 0.0
        for block_start in range(0, columns, BLOCK_SIZE):
            offsets = block_start + column_offsets
            mask = offsets < columns
            x1 = tl.load(x1_row_ptr + offsets * x1_col_stride, mask=mask, other=0.0)
            x2 = tl.load(x2_row_ptr + offsets * x2_col_stride, mask=mask, other=0.0)
            normalized_x1 = x1 / x1_scale
            normalized_x2 = x2 / x2_scale
            dot_sum += tl.sum(normalized_x1 * normalized_x2)
            x1_square_sum += tl.sum(normalized_x1 * normalized_x1)
            x2_square_sum += tl.sum(normalized_x2 * normalized_x2)

        similarity_denominator = tl.maximum(tl.sqrt_rn(x1_square_sum), eps_similarity) * tl.maximum(
            tl.sqrt_rn(x2_square_sum), eps_similarity
        )
        result = dot_sum / similarity_denominator
        tl.store(out_ptr + row, result.to(out_ptr.dtype.element_ty))


def build(context):
    def normalized_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        dim: int = 1,
        eps_similarity: float = 1e-8,
        p_norm: float = 2,
        eps_norm: float = 1e-12,
    ) -> torch.Tensor:
        if dim not in (1, -1, 0, -2) or x1.dim() != 2:
            raise ValueError("This kernel requires dim=1 (or dim=0) on a two-dimensional tensor")

        if dim in (0, -2):
            x1 = x1.transpose(0, 1)
            x2 = x2.transpose(0, 1)

        rows, columns = x1.shape
        output = torch.empty((rows,), device=x1.device, dtype=torch.float32)
        block_size = max(1, triton.next_power_of_2(columns))

        _normalized_cosine_similarity_kernel[(rows,)](
            x1,
            x2,
            output,
            columns,
            x1.stride(0),
            x1.stride(1),
            x2.stride(0),
            x2.stride(1),
            eps_similarity,
            eps_norm,
            P_NORM=float(p_norm),
            BLOCK_SIZE=block_size,
        )
        return output

    return normalized_cosine_similarity
