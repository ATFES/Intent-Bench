from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, channels, sequence, width) = (4, 4096, 4096, 4)
    x = torch.randn((batch, sequence, channels), device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn((channels, width), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((channels,), device='cuda', dtype=torch.bfloat16)
    source_module = load_module(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/convolution/causal_conv1d/causal_1d_conv.py', 'intent_v2_triton_causal_conv1d')
    source = functional_launch(lambda : source_module.causal_conv1d_fwd(x, weight, bias=bias, activation='silu'))
    return Side(source, Tolerance(atol=0.05, rtol=0.05), cuda_graph=True, owners=locals())
