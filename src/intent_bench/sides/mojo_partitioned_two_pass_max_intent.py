from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.reduction.two_pass import partitioned_max_partial, partitioned_max_reduce
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = -torch.rand((128, 257), dtype=torch.float32)
    partial = intent.compile(partitioned_max_partial, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    reduce = intent.compile(partitioned_max_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    state = {}

    def generated_launch():
        workspace = torch.empty((128, 300), dtype=torch.float32)
        partial(x, workspace)
        state['generated'] = reduce.run(workspace)
    return Side(PreparedLaunch(generated_launch, lambda : state['generated']), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='既有M128-N257-P300两阶段max，包含空子区域；原负输入、exact输出；单NUMA8核，完整host含workspace分配及两kernel同步；partial kernel完整写入Out，PyTorch单次amax为数学reference。', owners=locals())
