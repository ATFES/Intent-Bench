from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, IntentSequence
import intent
import torch
from intent.examples.kernels.ragged.grouped_gemm import ragged_grouped_gemm_backward_weight
from intent_bench.implementations.context import Tolerance

def prepare(context):
    left = torch.randn((3840, 4096), dtype=torch.float16)
    right = torch.randn_like(left)
    offsets = torch.tensor((0, 256, 768, 1792, 3840), dtype=torch.int32)
    generated = torch.empty((4, 4096, 4096), dtype=torch.float16)
    artifact = context.compile(ragged_grouped_gemm_backward_weight)
    sequence = IntentSequence()
    sequence.add(artifact, left, right, offsets, outputs=(generated,))
    call = sequence.prepared(lambda: generated, host_run=lambda: artifact(left, right, offsets, generated))
    return checked_call(call, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, device_type='cpu', note='既有 rows256/512/1024/2048、K4096/N4096、f16 grouped GEMM weight backward；f32 累加，完整分组native调用；双方输出预分配，PyTorch CPU reference，单 NUMA 8 核，沿用原容差。', owners=locals())
