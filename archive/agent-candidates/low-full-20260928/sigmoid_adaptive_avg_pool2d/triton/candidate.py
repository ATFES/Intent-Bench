import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    height,
    width,
    out_height,
    out_width,
    channels,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    output_plane = out_height * out_width
    channel = pid // output_plane
    spatial = pid - channel * output_plane
    out_y = spatial // out_width
    out_x = spatial - out_y * out_width

    start_y = (out_y * height) // out_height
    end_y = ((out_y + 1) * height + out_height - 1) // out_height
    start_x = (out_x * width) // out_width
    end_x = ((out_x + 1) * width + out_width - 1) // out_width

    offsets = tl.arange(0, BLOCK)
    window_width = end_x - start_x
    window_height = end_y - start_y
    window_size = window_width * window_height
    row = offsets // window_width
    col = offsets - row * window_width
    valid = offsets < window_size
    input_offset = (channel * height + start_y + row) * width + start_x + col
    values = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
    total = tl.sum(tl.where(valid, values, 0.0), axis=0)
    average = total / window_size
    result = 1.0 / (1.0 + tl.exp(-average))

    tl.store(out_ptr + pid, result)


def build(context):
    def sigmoid_adaptive_avg_pool2d(input, output_size):
        if isinstance(output_size, int):
            out_height = output_size
            out_width = output_size
        else:
            out_height, out_width = output_size

        n, channels, height, width = input.shape
        output = torch.empty(
            (n, channels, out_height, out_width),
            device=input.device,
            dtype=input.dtype,
        )
        grid = (n * channels * out_height * out_width,)
        _sigmoid_adaptive_avg_pool2d_kernel[grid](
            input,
            output,
            height,
            width,
            out_height,
            out_width,
            channels,
            BLOCK=256,
        )
        return output

    return sigmoid_adaptive_avg_pool2d
