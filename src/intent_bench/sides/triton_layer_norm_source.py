from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/triton/normalization/layer_norm/05-layer-norm_runtime.py', 'intent_v2_triton_layer_norm')
    source_function = runtime.load_layer_norm()
    source = functional_launch(lambda : source_function(x, (hidden,), weight, bias, 1e-05))
    return Side(source, Tolerance(atol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
