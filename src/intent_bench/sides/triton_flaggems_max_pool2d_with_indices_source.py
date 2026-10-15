from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8, 32, 128, 128), device='cuda', dtype=torch.float16)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flag-gems/vision/max_pool2d/max_pool2d_with_indices_runtime.py', 'intent_v2_triton_flaggems_max_pool_with_indices')
    source = functional_launch(lambda : runtime.upstream((x,)))
    return Side(source, (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=True, owners=locals())
