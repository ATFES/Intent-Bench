import torch
import intent
import intent.language as I


@intent.kernel
def _copy_matrix(
    src: I.In[I.f32, (32, 32)],
    dst: I.Out[I.f32, (32, 32)],
):
    rows = I.domain(0, 32)
    cols = I.domain(0, 32)
    dst[rows, cols] = src[rows, cols]


@intent.kernel
def _householder_qr_det(
    work: I.InOut[I.f32, (32, 32)],
    result: I.Out[I.f32, ()],
):
    qdet = I.cast(1.0, I.f32)
    rdet = I.cast(1.0, I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    two = I.cast(2.0, I.f32)
    minus_one = I.cast(-1.0, I.f32)

    for k in I.domain(0, 32):
        rows = I.domain(k, 32)
        coordinates = I.indices(rows)
        column = work[rows, k]
        leading = work[k, k]
        norm = I.sqrt(I.reduce.sum(column * column, axis=0, acc_dtype=I.f32))

        reflector_alpha = leading
        reflector_beta = zero
        if norm > zero:
            sign = I.select(leading >= zero, one, minus_one)
            reflector_alpha = -sign * norm
            vector = I.select(
                coordinates == k,
                leading - reflector_alpha,
                column,
            )
            vector_norm_squared = I.reduce.sum(
                vector * vector,
                axis=0,
                acc_dtype=I.f32,
            )
            reflector_beta = I.fdiv(two, vector_norm_squared)

            for j in I.domain(k, 32):
                target_column = work[rows, j]
                projection = I.reduce.sum(
                    vector * target_column,
                    axis=0,
                    acc_dtype=I.f32,
                )
                updated_column = target_column - reflector_beta * vector * projection
                work[rows, j] = updated_column

            qdet = qdet * minus_one

        rdet = rdet * reflector_alpha

    I.store(result, (), qdet * rdet)


def build(context):
    copy_artifact = context.compile("det_qr_copy", _copy_matrix)
    qr_artifact = context.compile("det_qr_householder", _householder_qr_det)

    def determinant_via_qr(A, *, mode="reduced", out=None):
        del mode
        if out is None:
            out = torch.empty((), device=A.device, dtype=A.dtype)
        workspace = torch.empty_like(A)
        copy_artifact(A, workspace)
        qr_artifact(workspace, out)
        return out

    return determinant_via_qr
