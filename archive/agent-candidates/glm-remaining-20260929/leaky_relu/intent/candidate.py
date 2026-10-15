from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def leaky_relu_out_of_place(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",), I.constraints(noalias=True)],
    negative_slope: I.f32,
) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        value = input[index]
        output[index] = I.select(
            value > I.cast(0.0, I.f32),
            value,
            value * negative_slope,
        )


@intent.kernel
def leaky_relu_in_place(
    data: I.InOut[I.f32, ("N",)],
    negative_slope: I.f32,
) -> None:
    for index in I.parallel(I.domain(0, data.shape[0])):
        value = data[index]
        data[index] = I.select(
            value > I.cast(0.0, I.f32),
            value,
            value * negative_slope,
        )


def build(context):
    out_of_place = context.compile(
        "leaky_relu_out_of_place",
        leaky_relu_out_of_place,
    )
    in_place = context.compile(
        "leaky_relu_in_place",
        leaky_relu_in_place,
    )

    def leaky_relu(input, negative_slope=0.01, inplace=False):
        if inplace:
            in_place(input, negative_slope)
            return input
        output = torch.empty_like(input)
        out_of_place(input, output, negative_slope)
        return output

    return leaky_relu
