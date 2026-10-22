from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    arguments = (x, weight, bias, 1.0 / hidden, 1e-05)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/flash-attention/normalization/layer_norm/layer_norm_runtime.py', 'intent_v2_triton_flash_layer_norm')
    source = functional_launch(lambda : runtime.upstream(arguments))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
