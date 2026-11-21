from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent.examples.kernels.normalization.dropout_residual_rms_norm import dropout_residual_rms_norm_backward_data, dropout_residual_rms_norm_forward
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    x,residual,weight,dnormalized,dresidual,*scalars = context.inputs
    forward_arguments = (x,residual,weight,*scalars)
    backward_arguments = context.inputs
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
