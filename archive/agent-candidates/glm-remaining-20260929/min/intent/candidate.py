import intent
import intent.language as I
import torch


GROUP_SIZE = 32


@intent.kernel
def _min_partial(
    input_tensor: I.In[I.f32, ("ROWS", "COLUMNS")],
    partial_values: I.Out[
        I.f32, ("PARTS", "OUTPUTS"), I.constraints(noalias=True)
    ],
    partial_indices: I.Out[
        I.i64, ("PARTS", "OUTPUTS"), I.constraints(noalias=True)
    ],
    GROUP_SIZE: I.Constexpr[int],
    REDUCE_AXIS: I.Constexpr[int],
):
    member_count = input_tensor.shape[0]
    if REDUCE_AXIS == 0:
        member_count = input_tensor.shape[0]
    else:
        member_count = input_tensor.shape[1]

    for group in I.parallel(I.domain(0, partial_values.shape[0])):
        start = group * GROUP_SIZE
        stop = start + GROUP_SIZE
        if stop > member_count:
            stop = member_count

        for output in I.parallel(I.domain(0, partial_values.shape[1])):
            best_value = I.cast(I.inf, I.f32)
            best_index = I.cast(0, I.i64)

            for member in I.domain(start, stop):
                if REDUCE_AXIS == 0:
                    candidate = input_tensor[member, output]
                else:
                    candidate = input_tensor[output, member]

                if candidate < best_value:
                    best_value = candidate
                    best_index = I.cast(member, I.i64)
                if (candidate != candidate) & (best_value == best_value):
                    best_value = candidate
                    best_index = I.cast(member, I.i64)

            partial_values[group, output] = best_value
            partial_indices[group, output] = best_index


@intent.kernel
def _combine_partials(
    partial_values: I.In[I.f32, ("PARTS", "OUTPUTS")],
    partial_indices: I.In[I.i64, ("PARTS", "OUTPUTS")],
    values: I.Out[I.f32, ("OUTPUTS",), I.constraints(noalias=True)],
    indices: I.Out[I.i64, ("OUTPUTS",), I.constraints(noalias=True)],
):
    for output in I.parallel(I.domain(0, values.shape[0])):
        best_value = partial_values[0, output]
        best_index = partial_indices[0, output]

        for group in I.domain(1, partial_values.shape[0]):
            candidate_value = partial_values[group, output]
            candidate_index = partial_indices[group, output]

            if candidate_value < best_value:
                best_value = candidate_value
                best_index = candidate_index
            if (candidate_value != candidate_value) & (
                best_value == best_value
            ):
                best_value = candidate_value
                best_index = candidate_index

        values[output] = best_value
        indices[output] = best_index


def build(context):
    partial_dim0 = context.compile(
        "min_partial_dim0",
        _min_partial,
        constexprs={"GROUP_SIZE": GROUP_SIZE, "REDUCE_AXIS": 0},
    )
    partial_dim1 = context.compile(
        "min_partial_dim1",
        _min_partial,
        constexprs={"GROUP_SIZE": GROUP_SIZE, "REDUCE_AXIS": 1},
    )
    combine = context.compile("min_combine_partials", _combine_partials)
    partial_cache = {}

    def min(input_tensor, dim=0, keepdim=False):
        if input_tensor.ndim != 2:
            raise ValueError("this specialization requires a 2-D input tensor")

        normalized_dim = dim
        if normalized_dim < 0:
            normalized_dim += input_tensor.ndim
        if normalized_dim not in (0, 1):
            raise ValueError("dimension out of range")

        rows, columns = input_tensor.shape
        if normalized_dim == 0:
            output_size = columns
        else:
            output_size = rows
        reduced_size = input_tensor.shape[normalized_dim]
        part_count = (reduced_size + GROUP_SIZE - 1) // GROUP_SIZE

        cache_key = (
            normalized_dim,
            rows,
            columns,
            part_count,
            str(input_tensor.device),
        )
        if cache_key not in partial_cache:
            partial_cache[cache_key] = (
                torch.empty(
                    (part_count, output_size),
                    dtype=torch.float32,
                    device=input_tensor.device,
                ),
                torch.empty(
                    (part_count, output_size),
                    dtype=torch.int64,
                    device=input_tensor.device,
                ),
            )
        partial_values, partial_indices = partial_cache[cache_key]
        values = torch.empty(
            (output_size,),
            dtype=torch.float32,
            device=input_tensor.device,
        )
        indices = torch.empty(
            (output_size,),
            dtype=torch.int64,
            device=input_tensor.device,
        )

        if normalized_dim == 0:
            partial_dim0(input_tensor, partial_values, partial_indices)
        else:
            partial_dim1(input_tensor, partial_values, partial_indices)
        combine(partial_values, partial_indices, values, indices)

        if keepdim:
            if normalized_dim == 0:
                values = values.view(1, columns)
                indices = indices.view(1, columns)
            else:
                values = values.view(rows, 1)
                indices = indices.view(rows, 1)

        return values, indices

    return min
