from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (4, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), dtype=torch.float16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), dtype=torch.float16)
    positions = torch.arange(sequence, dtype=torch.float32)
    inverse = 1.0 / 10000 ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().unsqueeze(0).to(torch.float16)
    sine = embedding.sin().unsqueeze(0).to(torch.float16)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_rope_qk')
    source_query = query_initial.clone()
    source_key = key_initial.clone()

    def source_launch():
        runtime.rope_qk(source_query, source_key, cosine, sine)

    def restore_source():
        source_query.copy_(query_initial)
        source_key.copy_(key_initial)
    source = PreparedCall(source_launch, lambda : (source_query, source_key), reset=restore_source)
    return checked_call(source, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, device_type='cpu', note='既有 rope_qk f16 InOut；PyTorch CPU reference，单 NUMA 8 核；每次调用前恢复 query/key，恢复不计时；双方计完整 host 调用。', owners=locals())
