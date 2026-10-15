from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.normalization.dropout_residual_rms_norm import dropout_residual_rms_norm_backward_data, dropout_residual_rms_norm_forward
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
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
    forward_arguments = (x, residual, weight, RMS_SEED, KEEP_PROBABILITY, inverse_keep_probability, inverse_features, epsilon, weight_offset)
    backward_arguments = (x, residual, weight, dnormalized, dresidual, RMS_SEED, KEEP_PROBABILITY, inverse_keep_probability, inverse_features, epsilon, weight_offset)
    forward = intent.compile(dropout_residual_rms_norm_forward, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    backward = intent.compile(dropout_residual_rms_norm_backward_data, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_outputs = {}

    def generated_launch():
        generated_forward = forward.run(*forward_arguments)
        generated_backward = backward.run(*backward_arguments)
        generated_outputs['output'] = (*generated_forward, *generated_backward)
    return Side(PreparedLaunch(generated_launch, lambda : generated_outputs['output']), tuple((Tolerance(atol=0.065) for _ in range(4))), cuda_graph=False, device_type='cpu', note='既有dropout residual RMSNorm forward/backward完整调用；bf16、seed17、keep0.9；CPU reference为NumPy Philox加PyTorch算术，调用内生成随机掩码，双方计完整host调用。', owners=locals())
