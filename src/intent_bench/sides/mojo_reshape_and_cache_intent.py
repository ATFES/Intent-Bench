from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.cache.reshape_and_cache import reshape_and_cache
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
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
    artifact = intent.compile(reshape_and_cache, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def generated_prepare():
        generated_key_cache.zero_()
        generated_value_cache.zero_()

    def generated_launch():
        artifact.run(key, value, slots, generated_key_cache, generated_value_cache)
    return Side(PreparedLaunch(generated_launch, lambda : (generated_key_cache, generated_value_cache), prepare=generated_prepare), (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有reshape-and-cache；T4096、H8、D128、2048x16 cache，随机i32 slot mapping；双f16 InOut cache独立恢复，完整host调用。', owners=locals())
