from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, length, width) = (64, 16384, 5)
    x = torch.randn((batch, length), device='cuda', dtype=torch.float16)
    weight = torch.randn((width,), device='cuda', dtype=torch.float16)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flag-gems/convolution/conv1d/conv1d_runtime.py', 'intent_v2_triton_flaggems_conv1d')
    source = functional_launch(lambda : runtime.upstream((x, weight)))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
