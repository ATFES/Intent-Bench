import torch
import intent
import intent.language as I


@intent.kernel
def min_dim0(
    input_tensor: I.In[I.f32, ("M", "N")],
    output_values: I.Out[I.f32, ("N",)],
    output_indices: I.Out[I.i64, ("N",)],
):
    M, N = input_tensor.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    negated = -input_tensor[rows, columns]
    maximums, argmaximums = I.arg_reduce.max(negated, axis=0, acc_dtype=I.f32)

    output_values[columns] = -maximums
    output_indices[columns] = I.cast(argmaximums, I.i64)


def build(context):
    compiled_min_dim0 = context.compile("min_dim0_float32", min_dim0, constexprs={})

    def min(input_tensor, dim, keepdim=False):
        output_size = input_tensor.shape[1]
        values = torch.empty(
            (output_size,), device=input_tensor.device, dtype=torch.float32
        )
        indices = torch.empty(
            (output_size,), device=input_tensor.device, dtype=torch.int64
        )
        compiled_min_dim0(input_tensor, values, indices)
        return values, indices

    return min
