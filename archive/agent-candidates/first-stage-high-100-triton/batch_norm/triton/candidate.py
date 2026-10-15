import torch
import triton
import triton.language as tl


@triton.jit
def _batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    eps,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    tiles_per_channel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_tile = pid // tiles_per_channel
    tile = pid % tiles_per_channel

    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial
    base = channel_tile * spatial + offsets

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    channel = channel_tile % channels
    mean = tl.load(mean_ptr + channel).to(tl.float32)
    variance = tl.load(var_ptr + channel).to(tl.float32)
    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)

    invstd = tl.rsqrt(variance + eps)
    scale = weight * invstd
    shift = bias - mean * scale
    y = x * scale + shift
    tl.store(y_ptr + base, y, mask=mask)


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    output = torch.empty_like(input)
    channels = input.shape[1]
    spatial = input.shape[2] * input.shape[3]
    block_size = 2048
    tiles_per_channel = (spatial + block_size - 1) // block_size
    grid = (input.shape[0] * channels * tiles_per_channel,)
    _batch_norm_inference_kernel[grid](
        input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        eps,
        channels=channels,
        spatial=spatial,
        tiles_per_channel=tiles_per_channel,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def build(context):
    return batch_norm
