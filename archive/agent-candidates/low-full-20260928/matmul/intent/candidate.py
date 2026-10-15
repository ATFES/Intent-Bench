import torch
import intent
import intent.language as I


@intent.kernel
def _matmul(
    tensor1: I.In[I.f16, ("M", "K")],
    tensor2: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, K = tensor1.shape
    _, N = tensor2.shape
    rows = I.domain(0, M)
    cols = I.domain(0, N)
    inner = I.domain(0, K)
    product = I.matmul(tensor1[rows, inner], tensor2[inner, cols], acc_dtype=I.f32)
    output[rows, cols] = I.cast(product, I.f16)


def build(context):
    matmul_artifact = context.compile("matmul_fp16", _matmul)

    def matmul(tensor1, tensor2):
        m, _ = tensor1.shape
        _, n = tensor2.shape
        output = torch.empty((m, n), device=tensor1.device, dtype=tensor1.dtype)
        matmul_artifact(tensor1, tensor2, output)
        return output

    return matmul
