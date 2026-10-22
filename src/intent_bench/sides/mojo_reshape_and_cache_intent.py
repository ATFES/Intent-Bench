from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.cache.reshape_and_cache import reshape_and_cache
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
BLOCKS = 2048
BLOCK_SIZE = 16
HEADS = 8
HEAD_DIMENSION = 128
TOKENS = 4096

def prepare(context):
    key = torch.randn((TOKENS, HEADS, HEAD_DIMENSION), dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.randperm(BLOCKS * BLOCK_SIZE, dtype=torch.int64)[:TOKENS].to(torch.int32)
    cache_shape = (BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIMENSION)
    generated_key_cache = torch.zeros(cache_shape, dtype=torch.float16)
    generated_value_cache = torch.zeros_like(generated_key_cache)
    artifact = context.compile(reshape_and_cache)

    def generated_prepare():
        generated_key_cache.zero_()
        generated_value_cache.zero_()

    generated = intent_call(artifact, (key, value, slots, generated_key_cache, generated_value_cache),
        observe=lambda _: (generated_key_cache, generated_value_cache), reset=generated_prepare)
    return checked_call(generated, (Tolerance(atol=0.0), Tolerance(atol=0.0)), device_type='cpu', note='Prepared reshape-and-cache; original T4096/H8/D128 and cache layout. Cache restoration outside timing.', owners=locals())
