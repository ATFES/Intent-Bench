from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import torch
import intent
from intent.examples.kernels.indexing.relations import scaled_index_add_unique
from intent_bench.implementations.context import Tolerance

def prepare(context):
    initial = torch.randn((65536, 1, 4096), dtype=torch.float16)
    source = torch.randn((32768, 1, 4096), dtype=torch.float16)
    indices = torch.arange(0, 65536, 2, dtype=torch.int64)
    scaling = torch.randn((4096,), dtype=torch.float16)
    artifact = context.compile(scaled_index_add_unique)

    def side():
        output = initial.clone()
        return intent_call(artifact, (output, indices, source, scaling, 1.0), reset=lambda: output.copy_(initial), observe=lambda _: output)
    return checked_call(side(), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='原65536x1x4096/32768唯一偶数indices f16 scaled-index-add，f32更新后f16写回；独立InOut状态每次恢复且恢复不计时；PyTorch CPU reference，原容差，prepared native调用。', owners=locals())
