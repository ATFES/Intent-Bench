from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def dense_matmul(
    lhs: I.In[I.f16, ("M", "K")],
    rhs: I.In[I.f16, ("K", "N")],
    product: I.Out[I.f16, ("M", "N")],
) -> None:
    result = I.matmul(lhs, rhs, acc_dtype=I.f32)
    product[:, :] = I.cast(result, I.f16)


def build(context):
    artifact = context.compile("dense_matmul_1024x1024x1024_f16", dense_matmul)

    def matmul(tensor1, tensor2, *, out=None):
        if out is None:
            return artifact.run(tensor1, tensor2)
        artifact(tensor1, tensor2, out)
        return out

    return matmul
