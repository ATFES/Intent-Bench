from __future__ import annotations
from intent_bench.sides import Side
from types import SimpleNamespace
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, heads, splits, dimension) = (8, 32, 16, 128)
    partial = torch.randn((batch, heads, splits, dimension), device='cuda', dtype=torch.bfloat16)
    partial_lse = torch.randn((batch, heads, splits), device='cuda', dtype=torch.float32)
    configs = tuple((SimpleNamespace(NUM_KV_SPLITS_POW2=1 << (splits - 1).bit_length(), TILE_D=tile, ACCESS_FORM=form, occupancy=occupancy, num_ctas=ctas) for tile in (32, 64, 128, 256) for form in (1, 2, 3) for occupancy in (1, 2) for ctas in (1, 2)))
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/flash_decode/splitk_reduce.py', 'splitk_attention_reduce')
    source_output = torch.empty((batch, heads, dimension), device='cuda', dtype=torch.bfloat16)

    def source_launch():
        source_module.splitk_reduce(partial, partial_lse, source_output, 8192, tuning_configs=configs, compiler_timeout=context.compiler_timeout_seconds)
    initial_launch(source_launch, side='source')
    source = PreparedLaunch(source_launch, lambda : source_output)
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
