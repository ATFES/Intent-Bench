from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (4, 32, 4096, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16)
    k = torch.randn(shape, device='cuda', dtype=torch.float16)
    v = torch.randn(shape, device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(dimension)
    capability_major = torch.cuda.get_device_capability()[0]
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/triton/attention/fused/06-fused-attention_runtime.py', 'intent_v2_triton_flash_attention')
    source_function = runtime.load_attention()
    # Match this tutorial's is_blackwell(): its warp-specialized path targets
    # SM100, not every architecture with a larger capability number (e.g. SM120).
    warp_specialize = capability_major == 10
    source = functional_launch(lambda : source_function(q, k, v, True, scale, warp_specialize))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, note='同算法；CUDA graph 设备执行计时；source 额外写 f32 M 辅助结果。', owners=locals())
