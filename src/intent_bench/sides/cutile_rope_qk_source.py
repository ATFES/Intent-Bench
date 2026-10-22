from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, sequence, query_heads, key_heads, dimension) = (2, 4096, 32, 8, 128)
    query = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    key = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    cosine = torch.randn((1, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    initial_query = query.clone()
    initial_key = key.clone()
    source_query = initial_query.clone()
    source_key = initial_key.clone()
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/position/rope/rope.py', 'rope_qk')

    def source_prepare():
        source_query.copy_(initial_query)
        source_key.copy_(initial_key)

    def source_launch():
        source_module.apply_rope_base(source_query, source_key, cosine, sine)
    source_prepare()
    initial_launch(source_launch, side='source')
    source_prepare()
    source = PreparedCall(source_launch, lambda : (source_query, source_key), source_prepare)
    return checked_call(source, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.02, rtol=0.01)), cuda_graph=True, note='Native cuTile RoPE; original bf16 intermediate rounding preserved. Input restoration outside timing.', owners=locals())
