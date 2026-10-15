import torch
import intent
import intent.language as I


@intent.kernel
def _fused(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C", "H", "W")],
    eps: I.f32,
    HAS_WEIGHT: I.Constexpr[bool],
    HAS_BIAS: I.Constexpr[bool],
):
    # The whole NCHW domain is independent; channel parameters are explicitly
    # reshaped so their broadcast axis is the channel axis.
    mean = I.reshape(running_mean, (1, "C", 1, 1))
    var = I.reshape(running_var, (1, "C", 1, 1))
    y = (x[:, :, :, :] - mean) * I.rsqrt(var + eps)
    if HAS_WEIGHT:
        y = y * I.reshape(weight, (1, "C", 1, 1))
    if HAS_BIAS:
        y = y + I.reshape(bias, (1, "C", 1, 1))
    y = y + I.cast(3.0, I.f32)
    y = I.minimum(I.maximum(y, I.cast(0.0, I.f32)), I.cast(6.0, I.f32))
    out[:, :, :, :] = y * I.cast(1.0 / 6.0, I.f32)


def build(context):
    artifacts = {}
    for has_weight in (False, True):
        for has_bias in (False, True):
            key = (has_weight, has_bias)
            artifacts[key] = context.compile(
                "fused_hardsigmoid_bn_%d_%d" % key,
                _fused,
                constexprs={"HAS_WEIGHT": has_weight, "HAS_BIAS": has_bias},
            )

    def fused_hardsigmoid_batch_norm(
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        inplace: bool = False,
    ) -> torch.Tensor:
        del training, momentum
        has_weight = weight is not None
        has_bias = bias is not None
        if weight is None:
            # The specialized kernel does not read this argument when absent.
            weight = running_mean
        if bias is None:
            bias = running_mean
        out = x if inplace else torch.empty_like(x)
        artifacts[(has_weight, has_bias)](
            x, running_mean, running_var, weight, bias, out, eps
        )
        return out

    return fused_hardsigmoid_batch_norm
