import torch
import intent
import intent.language as I


@intent.kernel
def asin_kernel(
    input_tensor: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    n, = input_tensor.shape
    points = I.domain(0, n)

    one = I.cast(1.0, I.f32)
    zero = I.cast(0.0, I.f32)
    half_pi = I.cast(1.5707963267948966, I.f32)

    value = I.cast(input_tensor[points], I.f32)
    magnitude = I.abs(value)
    endpoint_polynomial = (
        I.cast(1.5707288, I.f32)
        + magnitude
        * (
            I.cast(-0.2121144, I.f32)
            + magnitude
            * (
                I.cast(0.0742610, I.f32)
                + magnitude * I.cast(-0.0187293, I.f32)
            )
        )
    )
    endpoint = half_pi - I.sqrt(one - magnitude) * endpoint_polynomial
    signed_endpoint = I.select(value < zero, -endpoint, endpoint)
    result = I.select(value == zero, value, signed_endpoint)
    output[points] = I.cast(result, I.f16)


def build(context):
    artifact = context.compile("asin_intent_kernel", asin_kernel, constexprs={})

    def asin(input_tensor):
        output = torch.empty_like(input_tensor)
        artifact(input_tensor, output)
        return output

    return asin
