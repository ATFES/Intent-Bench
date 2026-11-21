from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent.examples.kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    arguments = context.inputs
    (_, generated) = compile_single(context, fused_add_rms_norm, arguments)
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
