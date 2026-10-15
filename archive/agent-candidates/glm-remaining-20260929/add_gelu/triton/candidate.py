import torch
import triton
import triton.language as tl


@triton.jit
def _add_gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    scalar_other,
    n_elements,
    input_stride,
    other_stride,
    output_stride,
    BLOCK: tl.constexpr,
    OTHER_IS_TENSOR: tl.constexpr,
    IS_TANH: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    input_offsets = offsets * input_stride
    output_offsets = offsets * output_stride

    input_value = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)
    if OTHER_IS_TENSOR:
        other_value = tl.load(
            other_ptr + offsets * other_stride, mask=valid, other=0.0
        )
    else:
        other_value = scalar_other

    x = input_value.to(tl.float32)
    y = other_value.to(tl.float32)
    z = x + alpha * y

    if not IS_TANH:
        cdf = 0.5 * (1.0 + tl.erf(z * 0.7071067811865476))
        result = z * cdf
    else:
        inner = 0.7978845608028654 * (
            z + 0.044715 * z * z * z
        )
        result = 0.5 * z * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(
        output_ptr + output_offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=valid,
    )


def build(context):
    def add_gelu(input, other, alpha=1, approximate="none", out=None):
        if input.dim() != 1:
            raise ValueError("add_gelu supports one-dimensional input")
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")

        count = input.numel()
        other_is_tensor = torch.is_tensor(other)
        if other_is_tensor:
            if other.dim() != 1 or other.numel() != count:
                raise ValueError("other must be scalar or have input's shape")

        if out is None:
            output = torch.empty_like(input)
        else:
            if out.dim() != 1 or out.numel() != count:
                raise ValueError("out must have input's shape")
            output = out

        if count == 0:
            return output

        if other_is_tensor:
            other_ptr = other
        else:
            other_ptr = input
            other = float(other)

        _add_gelu_kernel[(triton.cdiv(count, 2048),)](
            input,
            other_ptr,
            output,
            float(alpha),
            0.0 if other_is_tensor else other,
            count,
            input.stride(0),
            other.stride(0) if other_is_tensor else 0,
            output.stride(0),
            BLOCK=2048,
            OTHER_IS_TENSOR=other_is_tensor,
            IS_TANH=(approximate == "tanh"),
            num_warps=8,
        )
        return output

    return add_gelu
