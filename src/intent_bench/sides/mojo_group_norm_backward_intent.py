from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_sequence
import intent
import torch
from intent.examples.programs.composition import GroupNormBackward
from intent.examples.kernels.backward.group_norm import group_norm_backward_dx, group_norm_backward_weight_bias
from intent_bench.implementations.context import Tolerance
GROUP_NORM_BATCH = 32
GROUP_NORM_CHANNELS = 256
GROUP_NORM_CHANNELS_PER_GROUP = 8
GROUP_NORM_GROUPS = 32
GROUP_NORM_SPATIAL = 1024

def prepare(context):
    shape = (GROUP_NORM_BATCH, GROUP_NORM_CHANNELS, GROUP_NORM_SPATIAL)
    x = torch.randn(shape, dtype=torch.float16) * 0.5
    grad_y = torch.randn(shape, dtype=torch.float16) * 0.05
    weight = torch.randn((GROUP_NORM_CHANNELS,), dtype=torch.float16)
    grouped = x.float().reshape(GROUP_NORM_BATCH, GROUP_NORM_GROUPS, GROUP_NORM_CHANNELS_PER_GROUP, GROUP_NORM_SPATIAL)
    mean = grouped.mean(dim=(2, 3)).to(torch.float16)
    centered = grouped - mean.float()[:, :, None, None]
    rstd = torch.rsqrt(centered.square().mean(dim=(2, 3)) + 1e-05).to(torch.float16)
    inverse_group_elements = 1.0 / (GROUP_NORM_CHANNELS_PER_GROUP * GROUP_NORM_SPATIAL)
    dx_artifact = context.compile(group_norm_backward_dx)
    affine_artifact = context.compile(group_norm_backward_weight_bias)
    program = GroupNormBackward(dx_artifact, affine_artifact)
    call = program.prepare(x, grad_y, weight, mean, rstd, inverse_group_elements)
    generated = intent_sequence((call.dx, call.affine), call.result, host_run=call.launch)
    return checked_call(generated, (Tolerance(atol=0.02), Tolerance(atol=0.25), Tolerance(atol=0.25)), cuda_graph=False, device_type='cpu', note='既有B32-C256-S1024-G32 f16 GroupNorm backward；dx与weight/bias梯度两个kernel完整native序列，PyTorch CPU reference同算法。', owners=locals())
