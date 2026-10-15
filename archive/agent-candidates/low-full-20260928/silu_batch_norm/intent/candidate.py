import torch
import intent
import intent.language as I


@intent.kernel
def _eval(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    mean: I.In[I.f32, ("C",)],
    var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C", "H", "W")],
    eps: I.f32,
    HAS_WEIGHT: I.Constexpr[bool],
    HAS_BIAS: I.Constexpr[bool],
):
    N, C, H, W = x.shape
    for n in I.parallel(I.domain(0, N)):
        for c in I.parallel(I.domain(0, C)):
            m = mean[c]
            inv = I.fdiv(I.cast(1.0, I.f32), I.sqrt(var[c] + eps))
            a = weight[c] if HAS_WEIGHT else I.cast(1.0, I.f32)
            b = bias[c] if HAS_BIAS else I.cast(0.0, I.f32)
            for h in I.parallel(I.domain(0, H)):
                for w in I.parallel(I.domain(0, W)):
                    z = (x[n, c, h, w] - m) * inv * a + b
                    out[n, c, h, w] = z * I.fdiv(
                        I.cast(1.0, I.f32), I.cast(1.0, I.f32) + I.exp(-z)
                    )


def build(context):
    kernels = {}
    for has_weight in (False, True):
        for has_bias in (False, True):
            kernels[(has_weight, has_bias)] = context.compile(
                "silu_batch_norm_eval_%d%d" % (has_weight, has_bias),
                _eval,
                constexprs={"HAS_WEIGHT": has_weight, "HAS_BIAS": has_bias},
            )

    def silu_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        if training:
            raise NotImplementedError("training mode is outside the fixed profile")
        has_weight = weight is not None
        has_bias = bias is not None
        if not has_weight:
            weight = torch.empty_like(running_mean)
        if not has_bias:
            bias = torch.empty_like(running_mean)
        out = torch.empty_like(input)
        kernels[(has_weight, has_bias)](
            input, running_mean, running_var, weight, bias, out, eps
        )
        return out

    return silu_batch_norm
