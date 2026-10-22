from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
FEATURES = 4096
KEEP_PROBABILITY = 0.9
RMS_ROWS = 4096
RMS_SEED = 17

def prepare(context):
    inverse_features = 1.0 / FEATURES
    inverse_keep_probability = 1.0 / KEEP_PROBABILITY
    epsilon = 1e-06
    weight_offset = 1.0
    shape = (RMS_ROWS, FEATURES)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn((FEATURES,), dtype=torch.bfloat16)
    dnormalized = torch.randn_like(x) * 0.25
    dresidual = torch.randn_like(x) * 0.25
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_dropout_residual_rms_norm')
    source_outputs = {}

    def source_launch():
        source_outputs['output'] = runtime.dropout_residual_rms_norm(x, residual, weight, dnormalized, dresidual, RMS_SEED, KEEP_PROBABILITY, inverse_keep_probability, inverse_features, epsilon, weight_offset)
    return checked_call(PreparedCall(source_launch, lambda : source_outputs['output']), tuple((Tolerance(atol=0.065) for _ in range(4))), cuda_graph=False, device_type='cpu', note='既有dropout residual RMSNorm forward/backward完整调用；bf16、seed17、keep0.9；CPU reference为NumPy Philox加PyTorch算术，调用内生成随机掩码，双方计完整host调用。', owners=locals())
