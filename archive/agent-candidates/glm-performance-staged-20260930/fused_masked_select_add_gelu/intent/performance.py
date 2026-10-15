import torch
import intent
import intent.language as I


@intent.fn
def gelu(value: I.f32, USE_TANH: I.Constexpr[bool]) -> I.f32:
    if USE_TANH:
        cubic = value + 0.044715 * value * value * value
        activation = 1.0 + I.tanh(0.7978845608028654 * cubic, approximate=True)
    else:
        activation = 1.0 + I.erf(value / 1.4142135623730951)
    return 0.5 * value * activation


@intent.kernel
def count_groups(
    mask: I.In[I.bool, (1048576,)],
    counts: I.Out[I.i32, (512,)],
):
    grouped = I.reshape(I.cast(mask[:], I.i32), (512, 2048))
    counts[:] = I.reduce.sum(grouped, axis=1, acc_dtype=I.i32)


@intent.kernel
def compact_scalar(
    input: I.In[I.f32, (1048576,)],
    mask: I.In[I.bool, (1048576,)],
    counts: I.In[I.i32, (512,)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f32, ("K",)],
    USE_TANH: I.Constexpr[bool],
):
    for group in I.parallel(I.domain(0, 512)):
        groups = I.domain(0, 512)
        preceding = I.select(I.indices(groups) < group, counts[groups], 0)
        offset = I.reduce.sum(preceding, axis=0, acc_dtype=I.i32)
        lanes = I.domain(0, 2048)
        active = mask[group * 2048 + I.indices(lanes)]
        ranks = I.cumsum(I.cast(active, I.i32), axis=0, inclusive=False,
                         acc_dtype=I.i32)
        destinations = I.cast(offset + ranks, I.index)
        for lane in I.parallel(lanes):
            if active[lane]:
                value = input[group * 2048 + lane] + alpha * other
                output[destinations[lane]] = gelu(value, USE_TANH)


@intent.kernel
def compact_tensor(
    input: I.In[I.f32, (1048576,)],
    mask: I.In[I.bool, (1048576,)],
    counts: I.In[I.i32, (512,)],
    other: I.In[I.f32, ("K",)],
    alpha: I.f32,
    output: I.Out[I.f32, ("K",)],
    USE_TANH: I.Constexpr[bool],
):
    for group in I.parallel(I.domain(0, 512)):
        groups = I.domain(0, 512)
        preceding = I.select(I.indices(groups) < group, counts[groups], 0)
        offset = I.reduce.sum(preceding, axis=0, acc_dtype=I.i32)
        lanes = I.domain(0, 2048)
        active = mask[group * 2048 + I.indices(lanes)]
        ranks = I.cumsum(I.cast(active, I.i32), axis=0, inclusive=False,
                         acc_dtype=I.i32)
        destinations = I.cast(offset + ranks, I.index)
        for lane in I.parallel(lanes):
            if active[lane]:
                destination = destinations[lane]
                value = input[group * 2048 + lane] + alpha * other[destination]
                output[destination] = gelu(value, USE_TANH)


def build(context):
    counts_kernel = context.compile("count_groups", count_groups)
    scalar = {mode: context.compile("compact_scalar_" + str(mode), compact_scalar,
                                   constexprs={"USE_TANH": mode})
              for mode in (False, True)}
    tensor = {mode: context.compile("compact_tensor_" + str(mode), compact_tensor,
                                   constexprs={"USE_TANH": mode})
              for mode in (False, True)}

    def wrapper(input, mask, other, *, alpha=1, approximate="none", out=None):
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be none or tanh")
        if out is None:
            size = other.numel() if torch.is_tensor(other) else 524236
            out = torch.empty((size,), dtype=input.dtype, device=input.device)
        counts = torch.empty((512,), dtype=torch.int32, device=input.device)
        counts_kernel(mask, counts)
        if torch.is_tensor(other):
            tensor[approximate == "tanh"](input, mask, counts, other, alpha, out)
        else:
            scalar[approximate == "tanh"](input, mask, counts, other, alpha, out)
        return out

    return wrapper
