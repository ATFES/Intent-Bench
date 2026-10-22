from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/triton/normalization/softmax/02-fused-softmax_runtime.py', 'intent_v2_triton_fused_softmax')
    source_function = runtime.load_softmax()
    source = functional_launch(lambda : source_function(x))
    return checked_call(source, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
