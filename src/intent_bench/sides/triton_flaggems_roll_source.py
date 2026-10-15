from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/flag-gems/indexing/roll/roll_runtime.py', 'intent_v2_triton_flaggems_roll')
    source = functional_launch(lambda : runtime.upstream((x,)))
    return Side(source, Tolerance(atol=0.0), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
