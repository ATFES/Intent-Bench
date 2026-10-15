from __future__ import annotations

from typing import Sequence, Union

import intent
import intent.language as I


Size = Union[int, Sequence[int]]


@intent.kernel
def sigmoid_adaptive_avg_pool2d_kernel(
    input: I.In[I.f32, ("B", "C", "H", "W")],
    output: I.Out[I.f32, ("B", "C", 1, 1)],
):
    spatial_sum = I.reduce.sum(input, axis=(2, 3), acc_dtype=I.f32)
    spatial_count = I.cast(input.shape[2], I.f32) * I.cast(input.shape[3], I.f32)
    mean = I.fdiv(spatial_sum, spatial_count)
    output[:, :, 0, 0] = I.sigmoid(mean)


def _normalize_output_size(output_size: Size) -> tuple[int, int]:
    if isinstance(output_size, int):
        return (output_size, output_size)
    values = tuple(output_size)
    if len(values) != 2:
        raise ValueError("output_size must contain exactly two values")
    return (int(values[0]), int(values[1]))


def build(context):
    kernel = context.compile(
        "sigmoid_adaptive_avg_pool2d",
        sigmoid_adaptive_avg_pool2d_kernel,
    )

    def sigmoid_adaptive_avg_pool2d(
        input,
        output_size: Size = (1, 1),
    ):
        if input.ndim != 4:
            raise ValueError("input must be a 4D NCHW tensor")
        if input.dtype.name != "float32":
            raise ValueError("input must have dtype float32")
        if _normalize_output_size(output_size) != (1, 1):
            raise ValueError("the supplied profile requires output_size (1, 1)")

        output = input.new_empty((input.shape[0], input.shape[1], 1, 1))
        kernel(input, output)
        return output

    return sigmoid_adaptive_avg_pool2d
