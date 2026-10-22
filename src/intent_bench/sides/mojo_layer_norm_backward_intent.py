from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.backward.layer_norm import layer_norm_backward_reduce, layer_norm_backward_rows
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
LAYER_NORM_FEATURES = 4096
LAYER_NORM_PARTIAL_GROUPS = 128
LAYER_NORM_ROWS = 4096

def prepare(context):
    epsilon = 1e-05
    inverse_features = 1.0 / LAYER_NORM_FEATURES
    shape = (LAYER_NORM_ROWS, LAYER_NORM_FEATURES)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    dy = torch.randn(shape, dtype=torch.bfloat16) * 0.05
    weight = torch.randn((LAYER_NORM_FEATURES,), dtype=torch.bfloat16)
    x_f32 = x.float()
    mean = x_f32.mean(dim=1)
    centered = x_f32 - mean[:, None]
    rstd = torch.rsqrt(centered.square().mean(dim=1) + epsilon)
    rows_artifact = context.compile(layer_norm_backward_rows)
    reduce_artifact = context.compile(layer_norm_backward_reduce)
    dx = torch.empty_like(x)
    partial_shape = (LAYER_NORM_PARTIAL_GROUPS, LAYER_NORM_FEATURES)
    dw_partial = torch.zeros(partial_shape, dtype=torch.bfloat16)
    db_partial = torch.zeros_like(dw_partial)
    dw = torch.empty((LAYER_NORM_FEATURES,), dtype=torch.float32)
    db = torch.empty_like(dw)
    generated_state = {}

    def generated_launch():
        dw_partial.zero_()
        db_partial.zero_()
        rows_artifact(x, dy, weight, mean, rstd, dx, dw_partial, db_partial, inverse_features)
        reduce_artifact(dw_partial, db_partial, dw, db)
        generated_state['output'] = (dx, dw, db)
    return checked_call(PreparedCall(generated_launch, lambda : generated_state['output']), (Tolerance(atol=0.05), Tolerance(atol=0.125), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B4096-N4096 bf16 LayerNorm backward；rows partial后接reduce，完整返回dx/dweight/dbias，PyTorch CPU reference同算法。', owners=locals())
