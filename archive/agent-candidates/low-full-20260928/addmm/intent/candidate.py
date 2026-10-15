import torch
import intent
import intent.language as I


@intent.kernel
def addmm_kernel(
    input: I.In[I.f16, ("M", "N")],
    mat1: I.In[I.f16, ("M", "K")],
    mat2: I.In[I.f16, ("K", "N")],
    out: I.Out[I.f16, ("M", "N")],
    beta: I.f32,
    alpha: I.f32,
):
    # The free output coordinates are independent; matmul retains the K
    # contraction and the f32 accumulator before the f16 output conversion.
    product = I.matmul(mat1[:, :], mat2[:, :], acc_dtype=I.f32)
    if beta == 0.0:
        value = alpha * product
    else:
        value = beta * I.cast(input[:, :], I.f32) + alpha * product
    out[:, :] = I.cast(value, I.f16)


def build(context):
    kernel = context.compile("addmm_1024", addmm_kernel)

    def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
        if out is None:
            out = torch.empty((1024, 1024), device=mat1.device, dtype=mat1.dtype)
        kernel(input, mat1, mat2, out, float(beta), float(alpha))
        return out

    return addmm
