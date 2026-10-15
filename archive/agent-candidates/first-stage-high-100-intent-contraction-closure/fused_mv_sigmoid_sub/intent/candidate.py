import intent
import intent.language as I


@intent.kernel
def _fused_scalar_kernel(
    input: I.In[I.f32, (1024, 1024)],
    vec: I.In[I.f32, (1024,)],
    other: I.f32,
    alpha: I.f32,
    out: I.Out[I.f32, (1024,)],
):
    z = I.matvec(input, vec, acc_dtype=I.f32)
    s = I.sigmoid(z)
    rows = I.domain(0, 1024)
    out[rows] = s - alpha * other


def build(context):
    scalar_artifact = context.compile(
        "fused_mv_sigmoid_sub_scalar", _fused_scalar_kernel, constexprs={}
    )

    def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
        other_value = float(other)
        alpha_value = float(alpha)
        if out is None:
            return scalar_artifact.run(input, vec, other_value, alpha_value)
        scalar_artifact(input, vec, other_value, alpha_value, out)
        return out

    return fused_mv_sigmoid_sub
