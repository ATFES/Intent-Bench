from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def fused_inference_kernel(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    output: I.Out[I.f32, ("N", "C", "H", "W")],
    eps: I.f32,
) -> None:
    for batch in I.parallel(I.domain(0, x.shape[0])):
        for channel in I.parallel(I.domain(0, x.shape[1])):
            for row in I.parallel(I.domain(0, x.shape[2])):
                for column in I.parallel(I.domain(0, x.shape[3])):
                    normalized = (
                        (
                            x[batch, channel, row, column]
                            - running_mean[channel]
                        )
                        * I.rsqrt(running_var[channel] + eps)
                        * weight[channel]
                        + bias[channel]
                    )
                    clamped = I.minimum(
                        I.maximum(
                            normalized + I.cast(3.0, I.f32),
                            I.cast(0.0, I.f32),
                        ),
                        I.cast(6.0, I.f32),
                    )
                    output[batch, channel, row, column] = I.fdiv(
                        clamped, I.cast(6.0, I.f32)
                    )


def build(context):
    inference = context.compile("fused_hardsigmoid_batch_norm_inference", fused_inference_kernel)

    def fused_hardsigmoid_batch_norm(
        x,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
        inplace=False,
    ):
        if training:
            raise NotImplementedError("training execution is not compiled for the fixed inference profile")
        if weight is None or bias is None:
            raise NotImplementedError("default affine parameters are not compiled for the fixed inference profile")
        if inplace:
            raise NotImplementedError("in-place execution is not compiled for the fixed inference profile")
        return inference.run(x, running_mean, running_var, weight, bias, eps)

    return fused_hardsigmoid_batch_norm
