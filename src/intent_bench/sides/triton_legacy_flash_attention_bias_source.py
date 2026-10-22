from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (4, 32, 4096, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16) * 0.5
    k = torch.randn_like(q) * 0.5
    v = torch.randn_like(q) * 0.5
    bias = torch.randn((batch, heads, sequence), device='cuda', dtype=torch.float32) * 0.125
    scale = dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flash-attention/attention/fused/flash_attn_triton_runtime.py', 'intent_v2_triton_legacy_flash_attention')
    source_arguments = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), bias[:, :, None, :], scale)
    source = functional_launch(lambda : runtime.upstream(source_arguments))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, note='同算法；source 额外写 LSE 和临时数据。', owners=locals())
