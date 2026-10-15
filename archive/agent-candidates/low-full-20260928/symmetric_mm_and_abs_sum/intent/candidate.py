import torch
import intent
import intent.language as I


@intent.kernel
def _symmetric_mm_and_abs_sum(
    A: I.In[I.f32, ("N", "M")],
    C: I.InOut[I.f32, ("N", "N")],
    result: I.Out[I.f32, ()],
    alpha: I.f32,
    beta: I.f32,
):
    # The free N axes are independent output elements; M is the contraction axis.
    product = I.matmul(A, A, transpose_rhs=True, acc_dtype=I.f32)
    updated = alpha * product + beta * C[:, :]
    C[:, :] = updated
    result[()] = I.reduce.sum(I.abs(updated), axis=(0, 1))


def build(context):
    artifact = context.compile("symmetric_mm_and_abs_sum", _symmetric_mm_and_abs_sum)

    def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float = 1.0, beta: float = 0.5):
        result = torch.empty((), device=A.device, dtype=torch.float32)
        artifact(A, C, result, alpha, beta)
        return result

    return symmetric_mm_and_abs_sum
