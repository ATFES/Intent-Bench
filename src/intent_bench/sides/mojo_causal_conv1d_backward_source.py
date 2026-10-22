from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
BWD_CAUSAL_CONV_BATCH = 8
BWD_CAUSAL_CONV_CHANNELS = 2048
BWD_CAUSAL_CONV_LENGTH = 4096
BWD_CAUSAL_CONV_WIDTH = 4

def prepare(context):
    shape = (BWD_CAUSAL_CONV_BATCH, BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_LENGTH)
    x = torch.randn(shape, dtype=torch.float16) * 0.5
    weight = torch.randn((BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_WIDTH), dtype=torch.float16) * 0.1
    grad_output = torch.randn_like(x) * 0.05
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_causal_conv1d_backward')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.causal_conv1d_backward(x, weight, grad_output)
    return checked_call(PreparedCall(source_launch, lambda : source_state['output']), (Tolerance(atol=0.0025), Tolerance(atol=0.25), Tolerance(atol=0.25)), cuda_graph=False, device_type='cpu', note='既有B8-D2048-L4096-W4 f16 causal conv backward；partial kernel后接reduce kernel，PyTorch CPU reference同算法，双方完整host调用。', owners=locals())
