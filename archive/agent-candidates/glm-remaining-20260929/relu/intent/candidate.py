from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def relu_out_of_place(
    source: I.In[I.f32, ("N",)],
    destination: I.Out[I.f32, ("N",)],
) -> None:
    zero = I.cast(0.0, I.f32)
    for index in I.parallel(I.domain(0, source.shape[0])):
        destination[index] = I.maximum(zero, source[index])


@intent.kernel
def relu_in_place(tensor: I.InOut[I.f32, ("N",)]) -> None:
    zero = I.cast(0.0, I.f32)
    for index in I.parallel(I.domain(0, tensor.shape[0])):
        tensor[index] = I.maximum(zero, tensor[index])


def build(context):
    out_of_place = context.compile("relu_out_of_place", relu_out_of_place)
    in_place = context.compile("relu_in_place", relu_in_place)

    def relu(input, inplace: bool = False):
        if inplace:
            in_place(input)
            return input
        return out_of_place.run(input)

    return relu
