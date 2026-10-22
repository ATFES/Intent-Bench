from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (sequence, key_sequence, heads, topk, dimension, rope) = (2048, 4096, 64, 512, 128, 64)
    q_source = torch.randn((1, heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k_source = torch.randn((1, 1, key_sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v_source = torch.randn_like(k_source)
    qpe_source = torch.randn((1, heads, sequence, rope), device='cuda', dtype=torch.bfloat16)
    kpe_source = torch.randn((1, 1, key_sequence, rope), device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(sequence, device='cuda', dtype=torch.int32)[:, None]
    selected = torch.maximum(positions - torch.arange(topk, device='cuda', dtype=torch.int32)[None, :], torch.zeros((), device='cuda', dtype=torch.int32)).contiguous()
    source_indices = selected[None, :, None, :]
    scale = 1.0 / math.sqrt(dimension + rope)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/sparse_mla/sparse_mla.py', 'sparse_mla_prefill')

    def source_call():
        output = source_module.tile_sparse_mla(q_source, k_source, v_source, source_indices, qpe_source, kpe_source, is_causal=True, scaling=scale)
        return output[0].permute(1, 0, 2)
    source = functional_launch(source_call)
    return checked_call(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
