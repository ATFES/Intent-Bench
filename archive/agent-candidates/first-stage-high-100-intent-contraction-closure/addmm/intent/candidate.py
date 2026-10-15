import torch
import intent
import intent.language as I


@intent.kernel
def _addmm_kernel(
    input: I.In[I.f16, ("M", "N")],
    mat1: I.In[I.f16, ("M", "K")],
    mat2: I.In[I.f16, ("K", "N")],
    out: I.Out[I.f16, ("M", "N")],
    beta: I.f32,
    alpha: I.f32,
):
    M, N = input.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    product = I.matmul(mat1, mat2, acc_dtype=I.f32)
    scaled_product = alpha * product

    # Avoid reading input when beta is zero, matching addmm's non-propagation
    # contract for NaN and infinity in that operand.
    if beta == 0:
        result = scaled_product
    else:
        added = I.cast(input[rows, columns], I.f32)
        result = scaled_product + beta * added

    out[rows, columns] = I.cast(result, I.f16)


def build(context):
    addmm_artifact = context.compile("addmm_intent_kernel", _addmm_kernel)

    def addmm(
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        beta: float = 1,
        alpha: float = 1,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        addmm_artifact(input, mat1, mat2, out, beta, alpha)
        return out

    return addmm
