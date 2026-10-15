import collections

import torch

import intent
import intent.language as I


_CHUNK = 1024

_MaxResult = collections.namedtuple("_MaxResult", ["values", "indices"])


@intent.kernel
def partial_argmax(
    x: I.In[I.f32, ("N",)],
    part_max: I.Out[I.f32, ("P",)],
    part_idx: I.Out[I.i64, ("P",)],
    CHUNK: I.Constexpr[int],
):
    flat = x[I.domain(0, x.shape[0])]
    rows = I.reshape(flat, (part_max.shape[0], CHUNK))

    vals, locs = I.arg_reduce.max(rows, axis=1)

    p_dom = I.domain(0, part_max.shape[0])
    part_max[p_dom] = vals
    row_ids = I.indices(p_dom)
    part_idx[p_dom] = I.cast(row_ids * CHUNK + locs, I.i64)


@intent.kernel
def final_argmax(
    part_max: I.In[I.f32, ("P",)],
    part_idx: I.In[I.i64, ("P",)],
    out_val: I.Out[I.f32, ()],
    out_idx: I.Out[I.i64, ()],
):
    p_dom = I.domain(0, part_max.shape[0])
    vals = part_max[p_dom]
    ids = part_idx[p_dom]

    value, selected = I.arg_reduce.max(vals, axis=0)
    out_val[()] = value
    out_idx[()] = ids[selected]


@intent.kernel
def full_argmax(
    x: I.In[I.f32, ("N",)],
    out_val: I.Out[I.f32, ()],
    out_idx: I.Out[I.i64, ()],
):
    axis = I.domain(0, x.shape[0])
    value, local_idx = I.arg_reduce.max(x[axis], axis=0)
    out_val[()] = value
    out_idx[()] = I.cast(local_idx, I.i64)


def build(context):
    partial = context.compile(
        "partial_argmax", partial_argmax, constexprs={"CHUNK": _CHUNK}
    )
    final = context.compile("final_argmax", final_argmax)
    full = context.compile("full_argmax", full_argmax)

    def max(input, dim, keepdim=False):
        if input.dim() != 1:
            raise NotImplementedError("intent max is specialized for rank-1 inputs")

        if dim < 0:
            dim += input.dim()
        if dim != 0:
            raise NotImplementedError("intent max is specialized for dim 0")

        n = input.shape[0]
        out_val = torch.empty((), dtype=torch.float32, device=input.device)
        out_idx = torch.empty((), dtype=torch.int64, device=input.device)

        if n > 0 and n % _CHUNK == 0:
            p_count = n // _CHUNK
            part_max = torch.empty((p_count,), dtype=torch.float32, device=input.device)
            part_idx = torch.empty((p_count,), dtype=torch.int64, device=input.device)
            partial(input, part_max, part_idx)
            final(part_max, part_idx, out_val, out_idx)
        else:
            full(input, out_val, out_idx)

        if keepdim:
            out_val = out_val.reshape((1,))
            out_idx = out_idx.reshape((1,))

        return _MaxResult(out_val, out_idx)

    return max
