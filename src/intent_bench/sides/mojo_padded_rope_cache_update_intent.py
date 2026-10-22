from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
import intent
from intent.examples.kernels.position.rope_cache import padded_rope_cache_update as padded_rope_cache_update_definition
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call

def prepare(context):
    (batch, query_heads, kv_heads) = (32, 32, 8)
    (cache_length, dimension) = (8192, 128)
    padded_length = cache_length + 1
    packed_input = torch.randn((batch, query_heads + 2 * kv_heads, dimension), dtype=torch.float16)
    sequence_lengths = torch.full((batch,), padded_length, dtype=torch.int32)
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    storage_shape = (query_rows + 2 * cache_rows, dimension)
    artifact = context.compile(padded_rope_cache_update_definition)
    generated_storage = torch.empty(storage_shape, dtype=torch.float16)

    def generated_outputs():
        generated_query = generated_storage[:query_rows].view(batch, query_heads, dimension)
        generated_key_cache = generated_storage[query_rows:query_rows + cache_rows].view(batch, padded_length, kv_heads, dimension)
        generated_value_cache = generated_storage[query_rows + cache_rows:].view(batch, padded_length, kv_heads, dimension)
        return (generated_query, generated_key_cache[:, cache_length], generated_value_cache[:, cache_length])
    generated = intent_call(artifact, (packed_input, sequence_lengths, generated_storage, 10000.0, 1.0),
                            observe=lambda _: generated_outputs())
    return checked_call(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 padded RoPE cache f16 InOut；保持原 B32/QH32/KVH8/D128/cache8192 输入、query/key/value 三个输出和原 tolerances；默认计准备后的 native 执行；Python 调用仅在 host_call 模式计时，输出 storage 分配不计时。', owners=locals())
