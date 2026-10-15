import torch
import intent
import intent.language as I


@intent.kernel
def silu_batch_norm_kernel(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C", "H", "W"), I.constraints(noalias=True)],
    eps: I.f32,
    HAS_WEIGHT: I.Constexpr[bool],
    HAS_BIAS: I.Constexpr[bool],
):
    N, C, H, W = input.shape

    n = I.domain(0, N)
    c = I.domain(0, C)
    h = I.domain(0, H)
    w = I.domain(0, W)

    x = input[n, c, h, w]
    mean = I.reshape(running_mean[c], (1, C, 1, 1))
    var = I.reshape(running_var[c], (1, C, 1, 1))

    invstd = I.rsqrt(var + eps)
    y = (x - mean) * invstd

    if HAS_WEIGHT:
        gamma = I.reshape(weight[c], (1, C, 1, 1))
        y = y * gamma

    if HAS_BIAS:
        beta = I.reshape(bias[c], (1, C, 1, 1))
        y = y + beta

    out[n, c, h, w] = y * I.sigmoid(y)


def build(context):
    artifacts = {}
    for has_weight in (False, True):
        for has_bias in (False, True):
            name = f"silu_batch_norm_w{int(has_weight)}_b{int(has_bias)}"
            artifacts[(has_weight, has_bias)] = context.compile(
                name,
                silu_batch_norm_kernel,
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
        has_weight = weight is not None
        has_bias = bias is not None

        weight_view = weight if has_weight else running_mean
        bias_view = bias if has_bias else running_mean

        output = torch.empty_like(input)
        artifacts[(has_weight, has_bias)](
            input,
            running_mean,
            running_var,
            weight_view,
            bias_view,
            output,
            eps,
        )
        return output

    return silu_batch_norm
