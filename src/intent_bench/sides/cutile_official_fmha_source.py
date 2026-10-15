from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import official_source

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension) = (4, 32, 8, 4096, 128)
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    source_module = official_source(context, 'corpus/gpu/baselines/cutile/cutile-python/attention/fmha/AttentionFMHA_runtime.py', 'intent_v2_cutile_official_fmha')
    config = initial_launch(lambda : source_module['tune_cutile_fmha'](q, k, v, scale, query_group_size=query_heads // key_heads, causal=True), side='source')
    kernel = source_module['fmha_kernel'].replace_hints(num_ctas=config.num_ctas, occupancy=config.occupancy)
    source = functional_launch(lambda : source_module['cutile_fmha'](q, k, v, qk_scale=scale, query_group_size=query_heads // key_heads, causal=True, tile_m=config.TILE_M, tile_n=config.TILE_N, kernel=kernel))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
