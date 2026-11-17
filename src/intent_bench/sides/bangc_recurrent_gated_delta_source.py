from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source
from intent_bench.source_support import reference_launch

def prepare(context):
    from intent_bench.trees import to_device
    query, key, value, gate, beta, scale = to_device(context.inputs, 'cuda')
    source = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py', 'recurrent_gated_delta', needs_utils=True)
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return checked_call(reference_launch(lambda : source.recurrent_gated_delta_rule(query, key, value, gate, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=False)), (tolerance, tolerance), device_type='cuda', owners=locals())
