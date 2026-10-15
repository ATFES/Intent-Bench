from __future__ import annotations
from intent_bench.sides import Side
import torch
import intent
from kernels.indexing.relations import scaled_index_add_unique
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    initial = torch.randn((65536, 1, 4096), dtype=torch.float16)
    source = torch.randn((32768, 1, 4096), dtype=torch.float16)
    indices = torch.arange(0, 65536, 2, dtype=torch.int64)
    scaling = torch.randn((4096,), dtype=torch.float16)
    artifact = intent.compile(scaled_index_add_unique, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def side(function):
        output = initial.clone()
        return PreparedLaunch(lambda : function(output, indices, source, scaling, 1.0), lambda : output, prepare=lambda : output.copy_(initial))
    return Side(side(artifact.run), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='原65536x1x4096/32768唯一偶数indices f16 scaled-index-add，f32更新后f16写回；独立InOut状态每次恢复且恢复不计时；PyTorch CPU reference，原容差，完整host调用。', owners=locals())
