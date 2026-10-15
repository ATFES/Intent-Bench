from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.backward.causal_conv import causal_conv1d_backward_partials, causal_conv1d_backward_reduce
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
BWD_CAUSAL_CONV_BATCH = 8
BWD_CAUSAL_CONV_CHANNELS = 2048
BWD_CAUSAL_CONV_LENGTH = 4096
BWD_CAUSAL_CONV_WIDTH = 4

def prepare(context):
    shape = (BWD_CAUSAL_CONV_BATCH, BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_LENGTH)
    x = torch.randn(shape, dtype=torch.float16) * 0.5
    weight = torch.randn((BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_WIDTH), dtype=torch.float16) * 0.1
    grad_output = torch.randn_like(x) * 0.05
    partial = intent.compile(causal_conv1d_backward_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    reduce = intent.compile(causal_conv1d_backward_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_state = {}

    def generated_launch():
        (grad_x, partial_weight, partial_bias) = partial.run(x, weight, grad_output)
        (grad_weight, grad_bias) = reduce.run(partial_weight, partial_bias)
        generated_state['output'] = (grad_x, grad_weight, grad_bias)
    return Side(PreparedLaunch(generated_launch, lambda : generated_state['output']), (Tolerance(atol=0.0025), Tolerance(atol=0.25), Tolerance(atol=0.25)), cuda_graph=False, device_type='cpu', note='既有B8-D2048-L4096-W4 f16 causal conv backward；partial kernel后接reduce kernel，PyTorch CPU reference同算法，双方完整host调用。', owners=locals())
