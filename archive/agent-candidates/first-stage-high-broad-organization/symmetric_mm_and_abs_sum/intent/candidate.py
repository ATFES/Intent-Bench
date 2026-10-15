import torch
import intent
import intent.language as I


@intent.kernel
def _form_result(
    A: I.In[I.f32, ("N", "M")],
    C: I.In[I.f32, ("N", "N")],
    D: I.Out[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
):
    n, _ = A.shape
    rows = I.domain(0, n)
    cols = I.domain(0, n)

    product = I.matmul(A, A, acc_dtype=I.f32, transpose_rhs=True)
    D[rows, cols] = alpha * product + beta * C[rows, cols]


@intent.kernel
def _abs_sum(
    D: I.In[I.f32, ("N", "N")],
    result: I.Out[I.f32, ()],
):
    n, _ = D.shape
    rows = I.domain(0, n)
    cols = I.domain(0, n)
    values = I.abs(D[rows, cols])
    total = I.reduce.sum(values, axis=(0, 1), acc_dtype=I.f32)
    result[()] = total


def build(context):
    form_result = context.compile("symmetric_mm_form_result", _form_result)
    abs_sum = context.compile("symmetric_mm_abs_sum", _abs_sum)

    def symmetric_mm_and_abs_sum(A, C, alpha=1.0, beta=0.5):
        intermediate = torch.empty_like(C)
        result = torch.empty((), device=C.device, dtype=C.dtype)
        form_result(A, C, intermediate, alpha, beta)
        abs_sum(intermediate, result)
        return result

    return symmetric_mm_and_abs_sum
