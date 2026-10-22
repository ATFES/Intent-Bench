from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.backward.causal_conv import causal_conv1d_backward_partials, causal_conv1d_backward_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence
BWD_CAUSAL_CONV_BATCH = 8
BWD_CAUSAL_CONV_CHANNELS = 2048
BWD_CAUSAL_CONV_LENGTH = 4096
BWD_CAUSAL_CONV_WIDTH = 4

def prepare(context):
    shape = (BWD_CAUSAL_CONV_BATCH, BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_LENGTH)
    x = torch.randn(shape, dtype=torch.float16) * 0.5
    weight = torch.randn((BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_WIDTH), dtype=torch.float16) * 0.1
    grad_output = torch.randn_like(x) * 0.05
    partial = context.compile(causal_conv1d_backward_partials)
    reduce = context.compile(causal_conv1d_backward_reduce)
    generated_state = {}

    def generated_launch():
        (grad_x, partial_weight, partial_bias) = partial.run(x, weight, grad_output)
        (grad_weight, grad_bias) = reduce.run(partial_weight, partial_bias)
        generated_state['output'] = (grad_x, grad_weight, grad_bias)
    sequence = IntentSequence()
    grad_x, partial_weight, partial_bias = sequence.add(partial, x, weight, grad_output)
    grad_weight, grad_bias = sequence.add(reduce, partial_weight, partial_bias)
    call = sequence.prepared(lambda: (grad_x, grad_weight, grad_bias),
        host_run=generated_launch, host_result=lambda: generated_state['output'])
    return checked_call(call, (Tolerance(atol=0.0025), Tolerance(atol=0.25), Tolerance(atol=0.25)), device_type='cpu', note='Prepared complete B8/D2048/L4096/W4 f16 causal-convolution backward partial+reduce.', owners=locals())
