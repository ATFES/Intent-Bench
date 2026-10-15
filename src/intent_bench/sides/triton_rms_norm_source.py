from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    arguments = (x, weight, 1.0 / hidden, 1e-06)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/liger-kernel/normalization/rms_norm/rms_norm_runtime.py', 'intent_v2_triton_rms_norm')
    source = functional_launch(lambda : runtime.upstream(arguments))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
