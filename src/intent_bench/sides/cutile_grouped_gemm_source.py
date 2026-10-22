from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    rows = (256, 512, 1024, 2048)
    hidden = output = 4096
    group_x = tuple((torch.randn((row, hidden), device='cuda', dtype=torch.bfloat16) for row in rows))
    group_weight = tuple((torch.randn((hidden, output), device='cuda', dtype=torch.bfloat16) for _ in rows))
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/gemm/grouped/group_gemm.py', 'grouped_gemm')
    group_x_list = list(group_x)
    group_weight_list = list(group_weight)
    source = functional_launch(lambda : tuple(source_module.group_gemm(group_x_list, group_weight_list, static_persistent=True, transpose_b=False)))
    return checked_call(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 使用 host group shapes', owners=locals())
