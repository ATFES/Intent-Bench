from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def ifftshift_kernel(
    source: I.In[I.f32, ("N",)],
    destination: I.Out[I.f32, ("N",), I.constraints(noalias=True)],
) -> None:
    n = source.shape[0]
    half = n // 2
    for output_index in I.parallel(I.domain(0, n)):
        source_index = (output_index + half) % n
        destination[output_index] = source[source_index]


def build(context):
    kernel = context.compile("ifftshift_f32", ifftshift_kernel)

    def ifftshift(input, dim=None):
        return kernel.run(input)

    return ifftshift
