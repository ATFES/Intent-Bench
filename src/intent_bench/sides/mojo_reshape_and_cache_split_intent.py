from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.variants.decomposition import reshape_key_cache, reshape_value_cache
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
    (key_program, value_program) = (intent.compile(definition, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options) for definition in (reshape_key_cache, reshape_value_cache))

    def pipeline(key_cache, value_cache):
        key_program.run(key, slots, key_cache)
        value_program.run(value, slots, value_cache)

    def side(function):
        caches = tuple((torch.zeros((BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIMENSION), dtype=torch.float16) for _ in range(2)))

        def prepare():
            for cache in caches:
                cache.zero_()
        return PreparedLaunch(lambda : function(*caches), lambda : caches, prepare=prepare)
    return Side(side(pipeline), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='原T4096/H8/D128/2048x16 cache两kernel变体，随机唯一i32 slots，f16复制精确比较；双cache独立，清零不计时；计key/value两次kernel完整host调用。', owners=locals())
