from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _pooled_pairwise_distance_p2(
    x1: I.In[I.f32, ("B", "C", "OH", "WH", "OW", "WW")],
    x2: I.In[I.f32, ("B", "C", "OH", "WH", "OW", "WW")],
    distance: I.Out[I.f32, ("B",)],
    INV_WINDOW_SIZE: I.Constexpr[float],
    EPS: I.Constexpr[float],
) -> None:
    for batch_index in I.parallel(I.domain(0, x1.shape[0])):
        channels = I.domain(0, x1.shape[1])
        output_h = I.domain(0, x1.shape[2])
        window_h = I.domain(0, x1.shape[3])
        output_w = I.domain(0, x1.shape[4])
        window_w = I.domain(0, x1.shape[5])
        windows1 = x1[batch_index, channels, output_h, window_h, output_w, window_w]
        pooled1 = I.reduce.sum(
            windows1,
            axis=(1, 3),
            acc_dtype=I.f32,
        ) * I.cast(INV_WINDOW_SIZE, I.f32)

        windows2 = x2[batch_index, channels, output_h, window_h, output_w, window_w]
        pooled2 = I.reduce.sum(
            windows2,
            axis=(1, 3),
            acc_dtype=I.f32,
        ) * I.cast(INV_WINDOW_SIZE, I.f32)

        difference = pooled1 - pooled2
        squared_sum = I.reduce.sum(
            difference * difference,
            axis=(0, 1, 2),
            acc_dtype=I.f32,
        )
        distance[batch_index] = I.maximum(
            I.sqrt(squared_sum),
            I.cast(EPS, I.f32),
        )


def build(context):
    distance_artifact = context.compile(
        "fused_pooled_pairwise_distance_p2",
        _pooled_pairwise_distance_p2,
        constexprs={"INV_WINDOW_SIZE": 0.25, "EPS": 1.0e-6},
    )

    def fused_pairwise_distance_adaptive_avg_pool2d(
        x1,
        x2,
        output_size,
        p: float = 2.0,
        eps: float = 1.0e-6,
        keepdim: bool = False,
    ):
        if p != 2.0 or eps != 1.0e-6:
            raise ValueError("this specialization supports p=2.0 and eps=1e-6")

        if isinstance(output_size, int):
            output_h, output_w = output_size, output_size
        else:
            output_h, output_w = output_size

        batch_size, channels, height, width = x1.shape
        if x2.shape != x1.shape:
            raise ValueError("x1 and x2 must have the same shape")
        if output_h <= 0 or output_w <= 0:
            raise ValueError("output_size entries must be positive")
        if height % output_h != 0 or width % output_w != 0:
            raise ValueError("this specialization requires divisible adaptive windows")
        if x1.dtype is not x2.dtype or x1.dtype is not torch.float32:
            raise ValueError("x1 and x2 must be float32")

        window_h = height // output_h
        window_w = width // output_w
        if window_h * window_w != 4:
            raise ValueError("this specialization requires a four-element pooling window")

        windows1 = x1.view(batch_size, channels, output_h, window_h, output_w, window_w)
        windows2 = x2.view(batch_size, channels, output_h, window_h, output_w, window_w)
        distance = torch.empty((batch_size,), device=x1.device, dtype=x1.dtype)

        distance_artifact(windows1, windows2, distance)

        if keepdim:
            return distance.reshape((batch_size, 1))
        return distance

    return fused_pairwise_distance_adaptive_avg_pool2d
