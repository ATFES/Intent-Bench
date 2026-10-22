from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.routing.mhc import mhc_pre_gemm_sqrsum, mhc_pre_fuse
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    factors = 1 + torch.arange(4).mul(0.01).view(1, -1, 1)
    residual = (torch.randn((2048, 4, 4096), dtype=torch.float32) * factors).bfloat16()
    weight = (torch.randn((24, 4, 4096), dtype=torch.float32) * 0.0001 * factors).flatten(1, 2)
    scale = torch.randn((3,), dtype=torch.float32) * 0.1
    base = torch.randn((24,), dtype=torch.float32) * 0.1
    first = context.compile(mhc_pre_gemm_sqrsum)
    second = context.compile(mhc_pre_fuse, constexprs={'RMS_EPS': 1e-06, 'PRE_EPS': 1e-06, 'SINKHORN_EPS': 1e-06, 'POST_MULTIPLIER': 1.0, 'SINKHORN_REPEATS': 10})
    (generated, source) = ({}, {})

    def launch():
        (mixes, square_sum) = first.run(residual.flatten(1, 2), weight)
        generated['output'] = second.run(mixes, square_sum, scale, base, residual)
    sequence = IntentSequence()
    mixes, square_sum = sequence.add(first, residual.flatten(1, 2), weight)
    output = sequence.add(second, mixes, square_sum, scale, base, residual)
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, (Tolerance(atol=0.01, rtol=0.001), Tolerance(atol=0.01, rtol=0.001), Tolerance(atol=0.05, rtol=0.02)), device_type='cpu', note='Prepared T2048/H4096/streams4 MHC pre; original two kernels, precision, Sinkhorn10, epsilon and tolerances.', owners=locals())
