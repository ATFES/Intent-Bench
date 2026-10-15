from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flag-gems/scan/cumsum/cumsum_runtime.py', 'intent_v2_triton_flaggems_cumsum')
    source = functional_launch(lambda : runtime.upstream((x,)))
    return Side(source, Tolerance(atol=0.002, rtol=1e-05), cuda_graph=True, owners=locals())
