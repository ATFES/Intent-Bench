from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import cuda.tile as ct
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, heads, sequence, latent, rope, split_size) = (8, 64, 8192, 512, 64, 512)
    splits = sequence // split_size
    query = torch.randn((batch, heads, latent), device='cuda', dtype=torch.float16)
    query_rope = torch.randn((batch, heads, rope), device='cuda', dtype=torch.float16)
    cache = torch.randn((batch, sequence, latent), device='cuda', dtype=torch.float16)
    cache_rope = torch.randn((batch, sequence, rope), device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(latent + rope)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/mla_decode_split/mla_decoding_split_kv.py', 'splitk_mla_decode', needs_utils=True, needs_splitk=True)
    source_partial_output = torch.empty((batch, heads, splits, latent), device='cuda', dtype=torch.float16)
    source_partial_lse = torch.empty((batch, heads, splits), device='cuda', dtype=torch.float32)
    source_output = torch.empty_like(query)

    def source_launch():
        ct.launch(torch.cuda.current_stream(), ((heads + 15) // 16, batch, splits), source_module._naive_absorb_mla_transpose_kernel, (query, query_rope, cache, cache, cache_rope, source_partial_output, source_partial_lse, scale, batch, heads, sequence, splits, split_size, latent, 16, 128, rope, True))
        source_module.splitk_reduce(source_partial_output, source_partial_lse, source_output, sequence)
    initial_launch(source_launch, side='source')
    source = PreparedCall(source_launch, lambda : source_output)
    return checked_call(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
