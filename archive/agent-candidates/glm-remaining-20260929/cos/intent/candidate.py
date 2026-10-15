from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _cos_kernel(output: I.Out[I.f32, ("N",)], input_tensor: I.In[I.f32, ("N",)]) -> None:
    output[:] = I.cos(input_tensor[:])


def build(context):
    kernel = context.compile("cos_elementwise", _cos_kernel)

    def cos(input_tensor, *, out=None):
        if out is None:
            return kernel.run(input_tensor)
        kernel(out, input_tensor)
        return out

    return cos
