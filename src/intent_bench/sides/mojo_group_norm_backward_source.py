from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
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
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_group_norm_backward')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.group_norm_backward(x, grad_y, weight, mean, rstd)
    return Side(PreparedLaunch(source_launch, lambda : source_state['output']), (Tolerance(atol=0.02), Tolerance(atol=0.25), Tolerance(atol=0.25)), cuda_graph=False, device_type='cpu', note='既有B32-C256-S1024-G32 f16 GroupNorm backward；dx与weight/bias梯度两个kernel完整调用，PyTorch CPU reference同算法。', owners=locals())
