"""Independent Torch implementations for retained operator contracts.

These functions do not import or execute FlagGems. Stateful functions update the
supplied buffers; their workload adapters restore those buffers before each run.
"""
import torch
import torch.nn.functional as F

from ..corpus.cpu.baselines.pytorch.cpu_runtime import (
    adamw,
    batch_norm_training as _batch_norm_training,
    conv1d_same,
    group_norm_backward,
    softmax_backward,
    triangular_solve,
)


def addcmul(x, scale, bias):
    return torch.addcmul(bias[:, :, None], x, scale[:, :, None])


def batch_norm_training(*arguments):
    # The shared reference returns mutable buffers before its three outputs.
    return _batch_norm_training(*arguments)[2:]


def cumsum(x):
    return torch.cumsum(x, dim=-1)


def embedding(indices, table):
    return F.embedding(indices, table)


def fp8_mqa_logits(q, kv, kv_scale, head_weight, key_start, key_end):
    logits = (q.float() @ kv.float().T).relu()
    output = (logits * head_weight[..., None]).sum(dim=1) * kv_scale[None, :]
    keys = torch.arange(kv.shape[0], device=q.device)[None, :]
    return output.masked_fill((keys < key_start[:, None]) | (keys >= key_end[:, None]), -float("inf"))


def histogram(samples, output):
    output.add_(torch.histc(samples, bins=256, min=0, max=256).to(torch.int32))
    return output


def logsumexp(x):
    return torch.logsumexp(x, dim=-1)


def max_pool2d_with_indices(x):
    return F.max_pool2d(x, kernel_size=3, stride=2, padding=1, return_indices=True)


def roll(x):
    return torch.roll(x, shifts=-1, dims=0)


def transpose_copy(x):
    return x.T.contiguous()
