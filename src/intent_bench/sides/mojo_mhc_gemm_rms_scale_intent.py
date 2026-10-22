from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.routing.mhc import mhc_gemm_rms_partial, mhc_gemm_rms_finalize
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    x = torch.randn((2048, 16384), dtype=torch.bfloat16)
    weight = torch.randn((16384, 24), dtype=torch.bfloat16)
    bias = torch.randn((24,), dtype=torch.bfloat16)
    partial = context.compile(mhc_gemm_rms_partial, constexprs={'P': 16})
    finalize = context.compile(mhc_gemm_rms_finalize, constexprs={'P': 16, 'STREAMS': 4, 'ALPHA_PRE': 1.0, 'ALPHA_POST': 1.0, 'ALPHA_RESIDUAL': 1.0})
    (generated, source) = ({}, {})

    def launch():
        (linear, square_sum) = partial.run(x, weight)
        generated['output'] = finalize.run(linear, square_sum, bias, x.shape[1])
    sequence = IntentSequence()
    linear, square_sum = sequence.add(partial, x, weight)
    output = sequence.add(finalize, linear, square_sum, bias, x.shape[1])
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.005, rtol=0.001)), device_type='cpu', note='Prepared T2048/H4096/streams4/P16 MHC GEMM+RMS partial/finalize; original precision and tolerances.', owners=locals())
