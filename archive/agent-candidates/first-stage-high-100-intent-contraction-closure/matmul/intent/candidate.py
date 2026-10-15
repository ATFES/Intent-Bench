import torch
import intent
import intent.language as I


@intent.kernel
def matmul_kernel(
    tensor1: I.In[I.f16, ("M", "K")],
    tensor2: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, K = tensor1.shape
    _, N = tensor2.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    product = I.matmul(tensor1, tensor2, acc_dtype=I.f32)
    output[rows, columns] = I.cast(product, I.f16)


def build(context):
    artifact = context.compile("matmul_fp16", matmul_kernel, constexprs={})

    def wrapper(tensor1, tensor2):
        output = torch.empty(
            (tensor1.shape[0], tensor2.shape[1]),
            device=tensor1.device,
            dtype=torch.float16,
        )
        artifact(tensor1, tensor2, output)
        return output

    return wrapper
