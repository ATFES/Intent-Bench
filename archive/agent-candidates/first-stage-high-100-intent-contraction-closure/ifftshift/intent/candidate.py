import intent
import intent.language as I


@intent.kernel
def _ifftshift_1d(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    positions = I.domain(0, n)
    coordinates = I.indices(positions)
    source_coordinates = (coordinates + n // 2) % n
    output[positions] = input[source_coordinates]


def build(context):
    artifact = context.compile("ifftshift_1d", _ifftshift_1d)

    def ifftshift(input, dim=None):
        return artifact.run(input)

    return ifftshift
