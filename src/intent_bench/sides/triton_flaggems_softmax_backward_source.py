from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    probabilities = torch.randn((4096, 4097), device='cuda', dtype=torch.float32)
    gradient = torch.randn_like(probabilities)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/flag-gems/normalization/softmax/softmax_runtime.py', 'intent_v2_triton_flaggems_softmax_backward')
    source = functional_launch(lambda : runtime.upstream((probabilities, gradient)))
    return Side(source, Tolerance(atol=0.0001, rtol=1e-05), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
