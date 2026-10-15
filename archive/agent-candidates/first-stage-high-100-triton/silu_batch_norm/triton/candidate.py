import torch
import triton
import triton.language as tl


@triton.jit
def _silu_batch_norm_affine(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    channels,
    plane_size,
    eps,
    BLOCK: tl.constexpr,
):
    # The flattened input is N contiguous channel planes, each of plane_size
    # elements.  Keeping the channel index scalar makes the four parameter
    # loads uniform for the whole program.
    pid = tl.program_id(0)
    channel = pid % channels
    base = (pid // channels) * channels * plane_size + channel * plane_size
    offsets = base + tl.arange(0, BLOCK)

    x = tl.load(x_ptr + offsets)
    mean = tl.load(mean_ptr + channel)
    variance = tl.load(var_ptr + channel)
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)

    normalized = (x - mean) * tl.rsqrt(variance + eps)
    normalized = normalized * weight + bias
    result = normalized * (1.0 / (1.0 + tl.exp(-normalized)))
    tl.store(out_ptr + offsets, result)


def build(context):
    def wrapper(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        output = torch.empty_like(input)
        channels = input.shape[1]
        plane_size = input.shape[2] * input.shape[3]
        grid = (input.shape[0] * channels,)
        _silu_batch_norm_affine[grid](
            input,
            running_mean,
            running_var,
            weight,
            bias,
            output,
            channels,
            plane_size,
            eps,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
