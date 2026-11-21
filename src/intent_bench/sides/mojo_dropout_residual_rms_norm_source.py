from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    arguments = context.inputs
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_dropout_residual_rms_norm')
    source_outputs = {}

    def source_launch():
        source_outputs['output'] = runtime.dropout_residual_rms_norm(*arguments)
    return checked_call(PreparedCall(source_launch, lambda : source_outputs['output']), tuple((Tolerance(atol=0.065) for _ in range(4))), cuda_graph=False, device_type='cpu', note='既有dropout residual RMSNorm forward/backward完整调用；bf16、seed17、keep0.9；CPU reference为NumPy Philox加PyTorch算术，调用内生成随机掩码，双方计完整host调用。', owners=locals())
