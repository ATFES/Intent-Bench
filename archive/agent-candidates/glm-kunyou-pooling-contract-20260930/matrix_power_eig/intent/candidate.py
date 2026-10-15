import torch
import intent
import intent.language as I


@intent.kernel
def _spectral_reconstruct(v: I.In[I.f32, (256, 256)],
                          eigvals_pow: I.In[I.f32, (256,)],
                          result: I.Out[I.f32, (256, 256)]):
    scaled = v * I.reshape(eigvals_pow, (1, 256))
    result[:, :] = I.matmul(scaled, v, acc_dtype=I.f32, transpose_rhs=True)


def build(context):
    reconstruct_artifact = context.compile("spectral_reconstruct", _spectral_reconstruct)

    def matrix_power_eig(A: torch.Tensor, k: float, *, out: torch.Tensor = None) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(A, UPLO='L')
        eigvals_pow = eigvals ** k

        if out is not None:
            result = out
        else:
            result = torch.empty_like(A)

        reconstruct_artifact(eigvecs, eigvals_pow, result)
        return result

    return matrix_power_eig

