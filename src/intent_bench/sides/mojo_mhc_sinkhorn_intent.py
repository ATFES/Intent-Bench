from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.routing.mhc import mhc_sinkhorn
from intent_bench.implementations.context import Tolerance

def prepare(context):
    initial = torch.randn((8192, 4, 4), dtype=torch.float32)
    artifact = context.compile(mhc_sinkhorn)

    def side():
        values = initial.clone()
        return intent_call(artifact, (values,), reset=lambda: values.copy_(initial), observe=lambda _: values)
    return checked_call(side(), Tolerance(atol=0.0002, rtol=0.0001), cuda_graph=False, device_type='cpu', note='既有 T8192/streams4 f32 MHC Sinkhorn；exp 后固定 20 次行列归一化、无 epsilon；每次恢复独立输入且恢复不计时，PyTorch CPU reference、原容差，prepared native调用，单 NUMA 8 核。', owners=locals())
