import torch
import intent
import intent.language as I


_OUTPUT_SIZE = 524236


@intent.kernel
def _mask_prefix(
    mask: I.In[I.bool, ("N",)],
    prefix: I.Out[I.i32, ("N",)],
):
    domain = I.domain(0, mask.shape[0])
    counts = I.cast(mask[domain], I.i32)
    values = I.cumsum(counts, axis=0, inclusive=True, reverse=False, acc_dtype=I.i32)
    prefix[domain] = values


@intent.kernel
def _select_add_gelu(
    input: I.In[I.f32, ("N",)],
    mask: I.In[I.bool, ("N",)],
    prefix: I.In[I.i32, ("N",)],
    output: I.Out[I.f32, (_OUTPUT_SIZE,)],
    other: I.f32,
    alpha: I.f32,
    APPROXIMATE_TANH: I.Constexpr[bool],
):
    domain = I.domain(0, input.shape[0])
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)
    sqrt_two = I.cast(1.4142135623730951, I.f32)
    tanh_scale = I.cast(0.7978845608028654, I.f32)
    tanh_cubic = I.cast(0.044715, I.f32)

    for source_index in I.parallel(domain):
        selected = mask[source_index]
        if selected:
            destination_index = I.cast(prefix[source_index], I.index) - I.cast(1, I.index)
            if destination_index >= I.cast(0, I.index):
                if destination_index < I.cast(524236, I.index):
                    value = input[source_index]
                    shifted = value + alpha * other
                    if APPROXIMATE_TANH:
                        cubic = shifted * shifted * shifted
                        argument = tanh_scale * (shifted + tanh_cubic * cubic)
                        activation = I.tanh(argument, approximate=True)
                    else:
                        argument = I.fdiv(shifted, sqrt_two)
                        activation = I.erf(argument)
                    result = half * shifted * (one + activation)
                    I.store(output, destination_index, result)


@intent.kernel
def _copy_output(
    source: I.In[I.f32, (_OUTPUT_SIZE,)],
    destination: I.Out[I.f32, (_OUTPUT_SIZE,)],
):
    domain = I.domain(0, _OUTPUT_SIZE)
    destination[domain] = source[domain]


def build(context):
    prefix_kernel = context.compile("masked_select_gelu_prefix", _mask_prefix)
    exact_kernel = context.compile(
        "masked_select_add_gelu_exact",
        _select_add_gelu,
        constexprs={"APPROXIMATE_TANH": False},
    )
    tanh_kernel = context.compile(
        "masked_select_add_gelu_tanh",
        _select_add_gelu,
        constexprs={"APPROXIMATE_TANH": True},
    )
    copy_kernel = context.compile("masked_select_gelu_copy", _copy_output)

    def fused_masked_select_add_gelu(
        input,
        mask,
        other,
        *,
        alpha=1,
        approximate="none",
        out=None,
    ):
        prefix = torch.empty(
            (input.shape[0],),
            dtype=torch.int32,
            device=input.device,
        )
        prefix_kernel(mask, prefix)

        target = out
        temporary = None
        if target is None:
            target = torch.empty(
                (_OUTPUT_SIZE,),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            temporary = torch.empty(
                (_OUTPUT_SIZE,),
                dtype=input.dtype,
                device=input.device,
            )

        destination = temporary if temporary is not None else target
        if approximate == "tanh":
            tanh_kernel(input, mask, prefix, destination, other, alpha)
        else:
            exact_kernel(input, mask, prefix, destination, other, alpha)

        if temporary is not None:
            copy_kernel(temporary, target)
        return target

    return fused_masked_select_add_gelu
