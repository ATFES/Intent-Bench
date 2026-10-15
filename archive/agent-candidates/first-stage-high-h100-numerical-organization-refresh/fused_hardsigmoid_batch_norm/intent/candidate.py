import torch
import intent
import intent.language as I


def build(context):
    @intent.fn
    def apply_hardsigmoid(value):
        zero = I.cast(0.0, I.f32)
        three = I.cast(3.0, I.f32)
        six = I.cast(6.0, I.f32)
        shifted = value + three
        upper = I.select(shifted > six, six, shifted)
        clipped = I.select(upper < zero, zero, upper)
        return I.fdiv(clipped, six)

    @intent.kernel
    def fused_wb(
        x: I.In[I.f32, ("N", "C", "H", "W")],
        running_mean: I.In[I.f32, ("C",)],
        running_var: I.In[I.f32, ("C",)],
        weight: I.In[I.f32, ("C",)],
        bias: I.In[I.f32, ("C",)],
        out: I.Out[I.f32, ("N", "C", "H", "W")],
        eps: I.f32,
    ):
        n = I.domain(0, x.shape[0])
        c = I.domain(0, x.shape[1])
        h = I.domain(0, x.shape[2])
        w = I.domain(0, x.shape[3])
        value = x[n, c, h, w]
        mean = I.reshape(running_mean, (1, x.shape[1], 1, 1))
        variance = I.reshape(running_var, (1, x.shape[1], 1, 1))
        scale = I.reshape(weight, (1, x.shape[1], 1, 1))
        offset = I.reshape(bias, (1, x.shape[1], 1, 1))
        normalized = (value - mean) * I.rsqrt(variance + eps)
        activated = apply_hardsigmoid(normalized * scale + offset)
        out[n, c, h, w] = activated

    @intent.kernel
    def fused_w(
        x: I.In[I.f32, ("N", "C", "H", "W")],
        running_mean: I.In[I.f32, ("C",)],
        running_var: I.In[I.f32, ("C",)],
        weight: I.In[I.f32, ("C",)],
        out: I.Out[I.f32, ("N", "C", "H", "W")],
        eps: I.f32,
    ):
        n = I.domain(0, x.shape[0])
        c = I.domain(0, x.shape[1])
        h = I.domain(0, x.shape[2])
        w = I.domain(0, x.shape[3])
        value = x[n, c, h, w]
        mean = I.reshape(running_mean, (1, x.shape[1], 1, 1))
        variance = I.reshape(running_var, (1, x.shape[1], 1, 1))
        scale = I.reshape(weight, (1, x.shape[1], 1, 1))
        normalized = (value - mean) * I.rsqrt(variance + eps)
        activated = apply_hardsigmoid(normalized * scale)
        out[n, c, h, w] = activated

    @intent.kernel
    def fused_b(
        x: I.In[I.f32, ("N", "C", "H", "W")],
        running_mean: I.In[I.f32, ("C",)],
        running_var: I.In[I.f32, ("C",)],
        bias: I.In[I.f32, ("C",)],
        out: I.Out[I.f32, ("N", "C", "H", "W")],
        eps: I.f32,
    ):
        n = I.domain(0, x.shape[0])
        c = I.domain(0, x.shape[1])
        h = I.domain(0, x.shape[2])
        w = I.domain(0, x.shape[3])
        value = x[n, c, h, w]
        mean = I.reshape(running_mean, (1, x.shape[1], 1, 1))
        variance = I.reshape(running_var, (1, x.shape[1], 1, 1))
        offset = I.reshape(bias, (1, x.shape[1], 1, 1))
        normalized = (value - mean) * I.rsqrt(variance + eps)
        activated = apply_hardsigmoid(normalized + offset)
        out[n, c, h, w] = activated

    @intent.kernel
    def fused_none(
        x: I.In[I.f32, ("N", "C", "H", "W")],
        running_mean: I.In[I.f32, ("C",)],
        running_var: I.In[I.f32, ("C",)],
        out: I.Out[I.f32, ("N", "C", "H", "W")],
        eps: I.f32,
    ):
        n = I.domain(0, x.shape[0])
        c = I.domain(0, x.shape[1])
        h = I.domain(0, x.shape[2])
        w = I.domain(0, x.shape[3])
        value = x[n, c, h, w]
        mean = I.reshape(running_mean, (1, x.shape[1], 1, 1))
        variance = I.reshape(running_var, (1, x.shape[1], 1, 1))
        normalized = (value - mean) * I.rsqrt(variance + eps)
        activated = apply_hardsigmoid(normalized)
        out[n, c, h, w] = activated

    artifact_wb = context.compile("fused_hardsigmoid_batch_norm_wb", fused_wb)
    artifact_w = context.compile("fused_hardsigmoid_batch_norm_w", fused_w)
    artifact_b = context.compile("fused_hardsigmoid_batch_norm_b", fused_b)
    artifact_none = context.compile("fused_hardsigmoid_batch_norm_none", fused_none)

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
        if training:
            raise NotImplementedError("the supplied profile uses inference batch normalization")
        del momentum
        out = x if inplace else torch.empty_like(x)
        if weight is not None and bias is not None:
            artifact_wb(x, running_mean, running_var, weight, bias, out, eps)
        elif weight is not None:
            artifact_w(x, running_mean, running_var, weight, out, eps)
        elif bias is not None:
            artifact_b(x, running_mean, running_var, bias, out, eps)
        else:
            artifact_none(x, running_mean, running_var, out, eps)
        return out

    return fused_hardsigmoid_batch_norm
