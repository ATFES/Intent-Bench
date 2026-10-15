import torch
import intent
import intent.language as I


@intent.kernel
def _form_d(
    A: I.In[I.f32, ("M", "K")],
    C: I.In[I.f32, ("M", "M")],
    D: I.Out[I.f32, ("M", "M")],
    alpha: I.f32,
    beta: I.f32,
):
    product = I.matmul(A, A, acc_dtype=I.f32, transpose_rhs=True)
    D[:, :] = alpha * product + beta * C[:, :]


@intent.kernel
def _abs_sum(
    D: I.In[I.f32, ("M", "M")],
    result: I.Out[I.f32, ()],
):
    total = I.reduce.sum(I.abs(D[:, :]), axis=(0, 1))
    result[()] = total


def build(context):
    form_d = context.compile("symmetric_form_d", _form_d)
    abs_sum = context.compile("symmetric_abs_sum", _abs_sum)

    def symmetric_mm_and_abs_sum(A, C, alpha=1.0, beta=0.5):
        D = torch.empty((A.shape[0], A.shape[0]), dtype=torch.float32, device=A.device)
        result = torch.empty((), dtype=torch.float32, device=A.device)
        form_d(A, C, D, alpha, beta)
        abs_sum(D, result)
        return result

    return symmetric_mm_and_abs_sum
