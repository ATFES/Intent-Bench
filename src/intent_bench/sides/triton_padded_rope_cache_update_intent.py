from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.position.rope_cache import padded_rope_cache_update
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, kv_heads) = (32, 32, 8)
    (cache_length, dimension) = (8192, 128)
    padded_length = cache_length + 1
    query = torch.randn((batch, query_heads, dimension), device='cuda', dtype=torch.float16)
    key = torch.randn((batch, kv_heads, dimension), device='cuda', dtype=torch.float16)
    value = torch.randn_like(key)
    sequence_lengths = torch.full((batch,), padded_length, device='cuda', dtype=torch.int32)
    packed_input = torch.cat((query, key, value), dim=1)
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    generated_storage = torch.empty((query_rows + 2 * cache_rows, dimension), device='cuda', dtype=torch.float16)
    (_, generated_base) = compile_single(context, padded_rope_cache_update, (packed_input, sequence_lengths, generated_storage, 10000.0, 1.0))
    position = cache_length
    generated_query = generated_storage[:query_rows].view(batch, query_heads, dimension)
    generated_key_cache = generated_storage[query_rows:query_rows + cache_rows].view(batch, padded_length, kv_heads, dimension)
    generated_value_cache = generated_storage[query_rows + cache_rows:].view(batch, padded_length, kv_heads, dimension)
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : (generated_query, generated_key_cache[:, position], generated_value_cache[:, position]))
    return Side(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.0)), cuda_graph=False, owners=locals())
