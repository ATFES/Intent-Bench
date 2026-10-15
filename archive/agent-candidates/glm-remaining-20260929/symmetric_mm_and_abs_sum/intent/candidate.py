from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def symmetric_gemm_bias(
    A: I.In[I.f32, ("M", "K")],
    C: I.In[I.f32, ("M", "N")],
    D: I.Out[I.f32, ("M", "N"), I.constraints(noalias=True)],
    alpha: I.f32,
    beta: I.f32,
) -> None:
    rows = I.domain(0, A.shape[0])
    columns = I.domain(0, A.shape[0])
    reduction = I.domain(0, A.shape[1])

    left = A[rows, reduction]
    right = A[columns, reduction]
    product = I.matmul(left, right, transpose_rhs=True, acc_dtype=I.f32)
    D[rows, columns] = alpha * product + beta * C[rows, columns]


@intent.kernel
def abs_sum(
    D: I.In[I.f32, ("M", "N")],
    result: I.Out[I.f32, ()],
) -> None:
    D_value = D[:, :]
    result[()] = I.reduce.sum(I.abs(D_value), axis=(0, 1), acc_dtype=I.f32)


def build(context):
    gemm = context.compile("symmetric_mm_and_abs_sum_gemm", symmetric_gemm_bias)
    reduction = context.compile("symmetric_mm_and_abs_sum_abs_sum", abs_sum)

    def symmetric_mm_and_abs_sum(
        A: torch.Tensor,
        C: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.5,
    ) -> torch.Tensor:
        intermediate = torch.empty(
            (A.shape[0], C.shape[0]), dtype=A.dtype, device=A.device
        )
        result = torch.empty((), dtype=A.dtype, device=A.device)
        gemm(A, C, intermediate, alpha, beta)
        reduction(intermediate, result)
        return result

    return symmetric_mm_and_abs_sum
