from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, features, max_length) = (512, 128, 128)
    lengths = torch.arange(batch, device='cuda', dtype=torch.int32) % max_length + 1
    offsets = torch.empty(batch + 1, device='cuda', dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn(int(offsets[-1].item()), features, device='cuda', dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/tritonbench/ragged/jagged_mean/jagged_mean_runtime.py', 'intent_v2_triton_jagged_mean')
    source = functional_launch(lambda : runtime.upstream((values, offsets)))
    return checked_call(source, Tolerance(atol=2e-05, rtol=1e-05), cuda_graph=True, owners=locals())
