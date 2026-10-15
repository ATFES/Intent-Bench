from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = -torch.rand((128, 257), dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    state = {}

    def source_launch():
        state['source'] = runtime.partitioned_max(x)
    return Side(PreparedLaunch(source_launch, lambda : state['source']), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='既有M128-N257-P300两阶段max，包含空子区域；原负输入、exact输出；单NUMA8核，完整host含workspace分配及两kernel同步；partial kernel完整写入Out，PyTorch单次amax为数学reference。', owners=locals())
