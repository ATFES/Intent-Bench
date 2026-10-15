from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    shape = (64, 128, 4096)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], device='cuda', dtype=torch.bfloat16)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flag-gems/pointwise/addcmul/addcmul_runtime.py', 'intent_v2_triton_flaggems_addcmul')
    source = functional_launch(lambda : runtime.upstream((x, scale, bias)))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
