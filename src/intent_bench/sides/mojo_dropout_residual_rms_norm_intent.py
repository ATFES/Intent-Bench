from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.normalization.dropout_residual_rms_norm import dropout_residual_rms_norm_backward_data, dropout_residual_rms_norm_forward
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence
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
    forward = context.compile(dropout_residual_rms_norm_forward)
    backward = context.compile(dropout_residual_rms_norm_backward_data)
    generated_outputs = {}

    def generated_launch():
        generated_forward = forward.run(*forward_arguments)
        generated_backward = backward.run(*backward_arguments)
        generated_outputs['output'] = (*generated_forward, *generated_backward)
    sequence = IntentSequence()
    forward_output = sequence.add(forward, *forward_arguments)
    backward_output = sequence.add(backward, *backward_arguments)
    call = sequence.prepared(lambda: (*forward_output, *backward_output),
        host_run=generated_launch, host_result=lambda: generated_outputs['output'])
    return checked_call(call, tuple((Tolerance(atol=0.065) for _ in range(4))), device_type='cpu', note='Prepared dropout residual RMSNorm forward+backward; original bf16/seed17/keep0.9. Random-mask computation remains inside the operator.', owners=locals())
