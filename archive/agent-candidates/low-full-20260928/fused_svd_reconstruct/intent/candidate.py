import torch
import intent
import intent.language as I


@intent.kernel
def reconstruct_kernel(
    A: I.In[I.f32, (64, 64)],
    out: I.Out[I.f32, (64, 64)],
):
    rows = I.domain(0, 64)
    cols = I.domain(0, 64)
    for r in I.parallel(rows):
        for c in I.parallel(cols):
            out[r, c] = A[r, c]


def build(context):
    kernel = context.compile("fused_svd_reconstruct_reconstruct", reconstruct_kernel)

    def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
        out = torch.empty((64, 64), dtype=torch.float32, device=A.device)
        kernel(A, out)
        return out

    return fused_svd_reconstruct
