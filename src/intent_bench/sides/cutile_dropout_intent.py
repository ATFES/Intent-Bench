from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.regularization.dropout import xor_shift_dropout
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/regularization/dropout/dropout.py', 'dropout')
    seed = 12345
    probability = 0.1
    mixed_seed = source_module._mix_seed(seed)
    inverse_keep = 1.0 / (1.0 - probability)
    (_, generated) = compile_single(context, xor_shift_dropout, (x, mixed_seed, probability, inverse_keep))
    return Side(generated, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
