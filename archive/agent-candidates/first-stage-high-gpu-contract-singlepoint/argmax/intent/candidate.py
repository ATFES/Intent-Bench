import torch
import intent
import intent.language as I


@intent.kernel
def _partial_argmax(
    input_tensor: I.In[I.f16, ("N",)],
    partial_values: I.Out[I.f16, ("P",)],
    partial_indices: I.Out[I.i64, ("P",)],
    CHUNK: I.Constexpr[int],
):
    source = I.domain(0, input_tensor.shape[0])
    partitions = I.domain(0, partial_values.shape[0])

    for partition in I.parallel(partitions):
        begin = partition * CHUNK
        segment = source[begin : begin + CHUNK]
        values = input_tensor[segment]
        best_value, best_index = I.arg_reduce.max(values, axis=0)
        partial_values[partition] = best_value
        partial_indices[partition] = I.cast(
            I.cast(best_index, I.i64) + I.cast(begin, I.i64), I.i64
        )


@intent.kernel
def _final_argmax(
    partial_values: I.In[I.f16, ("P",)],
    partial_indices: I.In[I.i64, ("P",)],
    output: I.Out[I.i64, ()],
):
    best_value, partition = I.arg_reduce.max(partial_values, axis=0)
    output[()] = partial_indices[partition]


def build(context):
    partial = context.compile(
        "argmax_partial",
        _partial_argmax,
        constexprs={"CHUNK": 1024},
    )
    final = context.compile("argmax_final", _final_argmax, constexprs={})

    def argmax(input_tensor, dim, keepdim=False):
        num_partitions = (input_tensor.shape[0] + 1023) // 1024
        partial_values = torch.empty(
            (num_partitions,),
            dtype=torch.float16,
            device=input_tensor.device,
        )
        partial_indices = torch.empty(
            (num_partitions,),
            dtype=torch.int64,
            device=input_tensor.device,
        )
        output = torch.empty((), dtype=torch.int64, device=input_tensor.device)

        partial(input_tensor, partial_values, partial_indices)
        final(partial_values, partial_indices, output)
        return output

    return argmax
