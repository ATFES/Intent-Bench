from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    initial = torch.randn((65536, 1, 4096), dtype=torch.float16)
    source = torch.randn((32768, 1, 4096), dtype=torch.float16)
    indices = torch.arange(0, 65536, 2, dtype=torch.int64)
    scaling = torch.randn((4096,), dtype=torch.float16)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        output = initial.clone()
        return PreparedLaunch(lambda : function(output, indices, source, scaling, 1.0), lambda : output, prepare=lambda : output.copy_(initial))
    return Side(side(runtime.scaled_index_add), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='原65536x1x4096/32768唯一偶数indices f16 scaled-index-add，f32更新后f16写回；独立InOut状态每次恢复且恢复不计时；PyTorch CPU reference，原容差，完整host调用。', owners=locals())
