from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (tokens, hidden, streams) = (2048, 4096, 4)
    residual = torch.randn((tokens, streams, hidden), device='cuda', dtype=torch.bfloat16)
    layer_output = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    post_bf16 = torch.randn((tokens, streams), device='cuda', dtype=torch.bfloat16)
    residual_bf16 = torch.randn((tokens, streams, streams), device='cuda', dtype=torch.bfloat16)
    packed = torch.zeros((tokens, streams * (streams + 2)), device='cuda', dtype=torch.bfloat16)
    packed[:, streams:2 * streams] = post_bf16
    packed[:, 2 * streams:] = residual_bf16.view(tokens, -1)
    source_module = _source(context)
    source = functional_launch(lambda : source_module.mhc_apply_residual(residual.view(tokens, streams * hidden), layer_output, packed, streams).view(tokens, streams, hidden))
    return Side(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=True, note='同算法；source 保留 bf16 中间累加', owners=locals())

def _source(context):
    return tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/mhc/fused/mhc.py', 'mhc')
