from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    (batch, query_heads, kv_heads) = (32, 32, 8)
    (cache_length, dimension) = (8192, 128)
    padded_length = cache_length + 1
    packed_input = torch.randn((batch, query_heads + 2 * kv_heads, dimension), dtype=torch.float16)
    sequence_lengths = torch.full((batch,), padded_length, dtype=torch.int32)
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    storage_shape = (query_rows + 2 * cache_rows, dimension)
    generated_storage = torch.empty(storage_shape, dtype=torch.float16)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_padded_rope_cache_update')
    source_storage = torch.empty_like(generated_storage)

    def source_launch():
        runtime.padded_rope_cache_update(packed_input, sequence_lengths, source_storage, 10000.0, 1.0)

    def source_outputs():
        source_query = source_storage[:query_rows].view(batch, query_heads, dimension)
        source_key_cache = source_storage[query_rows:query_rows + cache_rows].view(batch, padded_length, kv_heads, dimension)
        source_value_cache = source_storage[query_rows + cache_rows:].view(batch, padded_length, kv_heads, dimension)
        return (source_query, source_key_cache[:, cache_length], source_value_cache[:, cache_length])
    source = PreparedCall(source_launch, source_outputs)
    return checked_call(source, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 padded RoPE cache f16 InOut；保持原 B32/QH32/KVH8/D128/cache8192 输入、query/key/value 三个输出和原 tolerances；双方计完整 host 调用，输出 storage 分配不计时。', owners=locals())
