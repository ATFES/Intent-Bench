from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
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
    source_key_cache = torch.zeros_like(generated_key_cache)
    source_value_cache = torch.zeros_like(generated_value_cache)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_reshape_and_cache')
    source_state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def source_prepare():
        source_key_cache.zero_()
        source_value_cache.zero_()

    def source_launch():
        source_state['output'] = runtime.reshape_and_cache(key, value, slots, source_key_cache, source_value_cache)
    return Side(PreparedLaunch(source_launch, lambda : source_state['output'], prepare=source_prepare), (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有reshape-and-cache；T4096、H8、D128、2048x16 cache，随机i32 slot mapping；双f16 InOut cache独立恢复，完整host调用。', owners=locals())
