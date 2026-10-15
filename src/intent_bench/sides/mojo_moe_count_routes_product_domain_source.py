from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    ids = torch.randint(0, 64, (4096, 2), dtype=torch.int32)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        counts = torch.zeros((64,), dtype=torch.int32)
        return PreparedLaunch(lambda : function(ids, counts), lambda : counts, prepare=lambda : counts.zero_())
    return Side(side(runtime.moe_count_routes), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='原product-domain MoE count T4096/top2/E64、随机i32 IDs，relaxed i32 atomic add；独立counts每次清零且清零不计时，CPU bincount参考，精确比较，完整host调用。', owners=locals())
