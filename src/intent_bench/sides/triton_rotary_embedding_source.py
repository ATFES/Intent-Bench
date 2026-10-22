from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, heads, dimension) = (4, 4096, 32, 128)
    x = torch.randn((batch, sequence, heads, dimension), device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(sequence, device='cuda', dtype=torch.float32)
    inverse = 1.0 / 10000 ** (torch.arange(0, dimension, 2, device='cuda', dtype=torch.float32) / dimension)
    angles = torch.outer(positions, inverse)
    (cosine, sine) = (angles.cos(), angles.sin())
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flash-attention/position/rotary/rotary_runtime.py', 'intent_v2_triton_rotary_embedding')
    source = functional_launch(lambda : runtime.source.apply_rotary(x, cosine, sine))
    return checked_call(source, Tolerance(atol=0.005, rtol=0.05), cuda_graph=True, owners=locals())
