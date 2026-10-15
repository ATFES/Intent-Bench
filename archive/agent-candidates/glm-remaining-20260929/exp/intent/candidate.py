from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def exp_elementwise(
    input_tensor: I.In[I.f32, ("N",)],
    output_tensor: I.Out[I.f32, ("N",)],
) -> None:
    for element_index in I.parallel(I.domain(0, input_tensor.shape[0])):
        output_tensor[element_index] = I.exp(input_tensor[element_index])


def build(context):
    exp_artifact = context.compile("exp_elementwise", exp_elementwise)

    def exp(input_tensor, out=None):
        if out is None:
            return exp_artifact.run(input_tensor)
        exp_artifact(input_tensor, out)
        return out

    return exp
