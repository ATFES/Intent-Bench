from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
LAYER_NORM_FEATURES = 4096
LAYER_NORM_ROWS = 4096

def prepare(context):
    epsilon = 1e-05
    shape = (LAYER_NORM_ROWS, LAYER_NORM_FEATURES)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    dy = torch.randn(shape, dtype=torch.bfloat16) * 0.05
    weight = torch.randn((LAYER_NORM_FEATURES,), dtype=torch.bfloat16)
    x_f32 = x.float()
    mean = x_f32.mean(dim=1)
    centered = x_f32 - mean[:, None]
    rstd = torch.rsqrt(centered.square().mean(dim=1) + epsilon)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_layer_norm_backward')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.layer_norm_backward(x, dy, weight, mean, rstd)
    return Side(PreparedLaunch(source_launch, lambda : source_state['output']), (Tolerance(atol=0.05), Tolerance(atol=0.125), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B4096-N4096 bf16 LayerNorm backward；rows partial后接reduce，完整返回dx/dweight/dbias，PyTorch CPU reference同算法。', owners=locals())
