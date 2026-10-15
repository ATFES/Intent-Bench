from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    shape = (8192, 14336)
    gate = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    up = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/liger-kernel/activation/swiglu/swiglu_runtime.py', 'intent_v2_triton_swiglu')
    source = functional_launch(lambda : runtime.upstream((gate, up)))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
