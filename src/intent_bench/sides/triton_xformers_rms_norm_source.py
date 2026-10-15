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
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/xformers/normalization/rms_norm/rmsnorm_kernels_runtime.py', 'intent_v2_triton_xformers_rms_norm')
    source_module = runtime.load_source()
    source = functional_launch(lambda : source_module._rms_norm_forward(x, weight, 1e-06))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
