from __future__ import annotations
from intent_bench.sides import Side
import torch
import intent
from kernels.position.rope_cache import padded_rope_cache_update as padded_rope_cache_update_definition
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (batch, query_heads, kv_heads) = (32, 32, 8)
    (cache_length, dimension) = (8192, 128)
    padded_length = cache_length + 1
    packed_input = torch.randn((batch, query_heads + 2 * kv_heads, dimension), dtype=torch.float16)
    sequence_lengths = torch.full((batch,), padded_length, dtype=torch.int32)
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    storage_shape = (query_rows + 2 * cache_rows, dimension)
    artifact = intent.compile(padded_rope_cache_update_definition, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_storage = torch.empty(storage_shape, dtype=torch.float16)

    def generated_launch():
        artifact.run(packed_input, sequence_lengths, generated_storage, 10000.0, 1.0)

    def generated_outputs():
        generated_query = generated_storage[:query_rows].view(batch, query_heads, dimension)
        generated_key_cache = generated_storage[query_rows:query_rows + cache_rows].view(batch, padded_length, kv_heads, dimension)
        generated_value_cache = generated_storage[query_rows + cache_rows:].view(batch, padded_length, kv_heads, dimension)
        return (generated_query, generated_key_cache[:, cache_length], generated_value_cache[:, cache_length])
    generated = PreparedLaunch(generated_launch, generated_outputs)
    return Side(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 padded RoPE cache f16 InOut；保持原 B32/QH32/KVH8/D128/cache8192 输入、query/key/value 三个输出和原 tolerances；双方计完整 host 调用，输出 storage 分配不计时。', owners=locals())
