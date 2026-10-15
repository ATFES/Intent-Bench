from __future__ import annotations
from intent_bench.sides import Side
from types import SimpleNamespace
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/regularization/dropout/dropout.py', 'dropout')
    seed = 12345
    probability = 0.1
    configs = tuple((SimpleNamespace(TILE_SIZE=1 << exponent, ACCESS_FORM=form) for exponent in range(5, 15) for form in (1, 2, 3)))
    source = functional_launch(lambda : source_module.dropout(x, seed, p=probability, training=True, inplace=False, tuning_configs=configs, compiler_timeout=context.compiler_timeout_seconds))
    return Side(source, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
