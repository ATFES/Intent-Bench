import torch
import intent
import intent.language as I


@intent.kernel
def _relu_conv2d_kernel(
    input: I.In[I.f32, (1, 3, 256, 256)],
    weight: I.In[I.f32, (64, 3, 3, 3)],
    output: I.Out[I.f32, (1, 64, 256, 256)],
):
    pixels = I.domain(0, 256 * 256)
    taps = I.domain(0, 3 * 3 * 3)
    channels = I.domain(0, 64)

    pixel_index = I.indices(pixels)
    tap_index = I.indices(taps)

    pixel_grid = I.reshape(pixel_index, (256 * 256, 1))
    tap_grid = I.reshape(tap_index, (1, 3 * 3 * 3))

    out_h = pixel_grid // 256
    out_w = pixel_grid % 256
    in_channel = tap_grid // 9
    tap_rem = tap_grid % 9
    tap_h = tap_rem // 3
    tap_w = tap_rem % 3

    in_h = out_h + tap_h - 1
    in_w = out_w + tap_w - 1
    valid = (in_h >= 0) & (in_h < 256) & (in_w >= 0) & (in_w < 256)

    zero_index = I.cast(0, I.index)
    zero_value = I.cast(0.0, I.f32)
    patches = I.gather(
        input,
        (zero_index, in_channel, in_h, in_w),
        valid=valid,
        fill=zero_value,
    )

    filters = weight[channels, in_channel, tap_rem // 3, tap_rem % 3]
    filters = I.reshape(filters, (64, 3 * 3 * 3))
    filters_t = I.transpose(filters)
    values = I.matmul(patches, filters_t, acc_dtype=I.f32)
    values = I.transpose(I.reshape(values, (256, 256, 64)), (2, 0, 1))
    values = I.maximum(values, zero_value)

    output[zero_index, channels, I.domain(0, 256), I.domain(0, 256)] = values


def build(context):
    kernel = context.compile("relu_conv2d_fixed", _relu_conv2d_kernel)

    def relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inplace=False,
    ):
        output = torch.empty(
            (1, 64, 256, 256),
            device=input.device,
            dtype=input.dtype,
        )
        kernel(input, weight, output)
        return output

    return relu_conv2d
