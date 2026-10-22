from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (4, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    positions = torch.arange(sequence, device='cuda', dtype=torch.float32)
    inverse = 1.0 / 10000 ** (torch.arange(0, dimension, 2, device='cuda', dtype=torch.float32) / dimension)
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().unsqueeze(0).to(torch.float16)
    sine = embedding.sin().unsqueeze(0).to(torch.float16)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/liger-kernel/position/rope/rope_runtime.py', 'intent_v2_triton_rope_qk_runtime')
    source_function = runtime.load_rope_forward()
    source_query = query_initial.clone()
    source_key = key_initial.clone()
    state: dict[str, object] = {}

    def source_launch():
        (q_out, k_out, _, _) = source_function(source_query, source_key, cosine, sine)
        state['outputs'] = (q_out, k_out)
    source_launch()
    source = PreparedCall(launch=source_launch, result=lambda : state['outputs'], reset=lambda : (source_query.copy_(query_initial), source_key.copy_(key_initial)))
    return checked_call(source, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, owners=locals())
