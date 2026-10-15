import intent
import intent.language as I


@intent.kernel
def _tensordot_kernel(
    a: I.In[I.f16, (1024, 8, 8)],
    b: I.In[I.f16, (8, 8, 1024)],
    out: I.Out[I.f16, (1024, 1024)],
):
    a_matrix = I.reshape(a, (1024, 64))
    b_matrix = I.reshape(b, (64, 1024))
    result = I.matmul(a_matrix, b_matrix, acc_dtype=I.f32)
    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)
    out[rows, columns] = I.cast(result, I.f16)


def build(context):
    artifact = context.compile("tensordot_1024x8x8", _tensordot_kernel)

    def tensordot(a, b, dims):
        return artifact.run(a, b)

    return tensordot
