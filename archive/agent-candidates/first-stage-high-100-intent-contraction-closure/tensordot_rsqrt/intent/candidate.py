import intent
import intent.language as I


@intent.kernel
def _tensordot_rsqrt_kernel(
    a: I.In[I.f32, ("M", "K")],
    b: I.In[I.f32, ("K", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    M, K = a.shape
    _, N = b.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)

    product = I.matmul(
        a[rows, reduction],
        b[reduction, columns],
        acc_dtype=I.f32,
    )
    out[rows, columns] = I.rsqrt(product)


def build(context):
    artifact = context.compile(
        "tensordot_rsqrt_kernel",
        _tensordot_rsqrt_kernel,
        constexprs={},
    )

    def tensordot_rsqrt(a, b, dims):
        return artifact.run(a, b)

    return tensordot_rsqrt
