import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    channels = offsets % CHANNELS

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(running_mean_ptr + channels, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + channels, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0)

    normalized = (x - mean) * tl.rsqrt(var + eps)
    value = normalized * weight + bias
    output = tl.sigmoid(value)
    tl.store(output_ptr + offsets, output, mask=mask)


def build(context):
    def wrapper(input, running_mean, running_var, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5):
        output = torch.empty_like(input)
        n_elements = input.numel()
        channels = input.shape[1]
        grid = (triton.cdiv(n_elements, 1024),)
        _sigmoid_batch_norm_kernel[grid](
            input,
            running_mean,
            running_var,
            weight,
            bias,
            output,
            n_elements,
            eps,
            CHANNELS=channels,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
