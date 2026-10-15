from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, channels, spatial, groups) = (32, 256, 1024, 32)
    x = torch.randn((batch, channels, spatial), device='cuda', dtype=torch.float16)
    grad_y = torch.randn_like(x)
    weight = torch.randn((channels,), device='cuda', dtype=torch.float16)
    mean = torch.randn((batch, groups), device='cuda', dtype=torch.float16)
    rstd = torch.rand((batch, groups), device='cuda', dtype=torch.float16) + 0.5
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/flag-gems/normalization/group_norm/groupnorm_runtime.py', 'intent_v2_triton_flaggems_group_norm_backward')
    source = functional_launch(lambda : runtime.upstream((x, grad_y, weight, mean, rstd)))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
