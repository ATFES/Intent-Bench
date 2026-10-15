from __future__ import annotations

import intent
import intent.language as I
import torch


_OUTPUT_SIZE = 524236


@intent.kernel
def _masked_add_gelu_scalar_other(
    input: I.In[I.f32, ("N",)],
    mask: I.In[I.bool, ("N",)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f32, ("K",)],
    USE_TANH: I.Constexpr[bool],
) -> None:
    active = mask[:]
    selected = I.cast(active, I.i32)
    inclusive = I.cumsum(selected, axis=0, acc_dtype=I.i32)
    destinations = inclusive - selected
    for source_position in I.parallel(I.domain(0, input.shape[0])):
        if mask[source_position]:
            value = input[source_position] + alpha * other
            if USE_TANH:
                cubic = value + I.cast(0.044715, I.f32) * value * value * value
                activation = I.tanh(
                    I.cast(0.7978845608028654, I.f32) * cubic,
                    approximate=True,
                )
            else:
                activation = 1.0 + I.erf(
                    I.fdiv(value, I.cast(1.4142135623730951, I.f32))
                )
            result = I.cast(0.5, I.f32) * value * activation
            output[I.cast(destinations[source_position], I.index)] = result


@intent.kernel
def _masked_add_gelu_tensor_other(
    input: I.In[I.f32, ("N",)],
    mask: I.In[I.bool, ("N",)],
    other: I.In[I.f32, ("K",)],
    alpha: I.f32,
    output: I.Out[I.f32, ("K",)],
    USE_TANH: I.Constexpr[bool],
) -> None:
    active = mask[:]
    selected = I.cast(active, I.i32)
    inclusive = I.cumsum(selected, axis=0, acc_dtype=I.i32)
    destinations = inclusive - selected
    for source_position in I.parallel(I.domain(0, input.shape[0])):
        if mask[source_position]:
            destination = I.cast(destinations[source_position], I.index)
            value = input[source_position] + alpha * other[destination]
            if USE_TANH:
                cubic = value + I.cast(0.044715, I.f32) * value * value * value
                activation = I.tanh(
                    I.cast(0.7978845608028654, I.f32) * cubic,
                    approximate=True,
                )
            else:
                activation = 1.0 + I.erf(
                    I.fdiv(value, I.cast(1.4142135623730951, I.f32))
                )
            result = I.cast(0.5, I.f32) * value * activation
            output[destination] = result


def build(context):
    scalar_kernels = {
        False: context.compile(
            "masked_add_gelu_scalar_exact",
            _masked_add_gelu_scalar_other,
            constexprs={"USE_TANH": False},
        ),
        True: context.compile(
            "masked_add_gelu_scalar_tanh",
            _masked_add_gelu_scalar_other,
            constexprs={"USE_TANH": True},
        ),
    }
    tensor_kernels = {
        False: context.compile(
            "masked_add_gelu_tensor_exact",
            _masked_add_gelu_tensor_other,
            constexprs={"USE_TANH": False},
        ),
        True: context.compile(
            "masked_add_gelu_tensor_tanh",
            _masked_add_gelu_tensor_other,
            constexprs={"USE_TANH": True},
        ),
    }

    def fused_masked_select_add_gelu(
        input,
        mask,
        other,
        *,
        alpha=1,
        approximate="none",
        out=None,
    ):
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")
        if input.shape != mask.shape:
            raise ValueError("input and mask must have the same shape")
        if input.ndim != 1 or mask.ndim != 1:
            raise ValueError("this implementation requires one-dimensional inputs")
        if input.dtype != torch.float32 or out_is_wrong_dtype(out):
            raise ValueError("input and output must be float32")

        destination_count = _OUTPUT_SIZE if not torch.is_tensor(other) else other.numel()
        if out is None:
            output = torch.empty(
                destination_count,
                dtype=torch.float32,
                device=input.device,
            )
        else:
            if out.shape != (destination_count,):
                raise ValueError("output has an incompatible shape")
            output = out

        use_tanh = approximate == "tanh"
        if torch.is_tensor(other):
            if other.dtype != torch.float32 or other.ndim != 1:
                raise ValueError("tensor other must be one-dimensional float32")
            if other.shape != output.shape:
                raise ValueError("other must be broadcastable to the selected shape")
            tensor_kernels[use_tanh](
                input,
                mask,
                other,
                alpha,
                output,
            )
        else:
            scalar_kernels[use_tanh](
                input,
                mask,
                other,
                alpha,
                output,
            )
        return output

    return fused_masked_select_add_gelu


def out_is_wrong_dtype(out):
    return out is not None and out.dtype != torch.float32
