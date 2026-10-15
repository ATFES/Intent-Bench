from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from kernels.streaming.mla import token_sparse_mla_value_prefill
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (sequence, key_sequence, heads, topk, dimension, rope) = (2048, 4096, 64, 512, 128, 64)
    q_source = torch.randn((1, heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k_source = torch.randn((1, 1, key_sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v_source = torch.randn_like(k_source)
    qpe_source = torch.randn((1, heads, sequence, rope), device='cuda', dtype=torch.bfloat16)
    kpe_source = torch.randn((1, 1, key_sequence, rope), device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(sequence, device='cuda', dtype=torch.int32)[:, None]
    selected = torch.maximum(positions - torch.arange(topk, device='cuda', dtype=torch.int32)[None, :], torch.zeros((), device='cuda', dtype=torch.int32)).contiguous()
    q = q_source[0].permute(1, 0, 2).contiguous()
    qpe = qpe_source[0].permute(1, 0, 2).contiguous()
    k = k_source[0, 0]
    v = v_source[0, 0]
    kpe = kpe_source[0, 0]
    scale = 1.0 / math.sqrt(dimension + rope)
    (_, generated) = compile_single(context, token_sparse_mla_value_prefill, (q, qpe, k, v, kpe, selected, scale))
    return Side(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
