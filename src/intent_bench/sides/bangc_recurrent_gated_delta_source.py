from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source
from intent_bench.source_support import reference_launch

def prepare(context):
    arguments = _gated_delta_inputs()
    source = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py', 'recurrent_gated_delta', needs_utils=True)
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return Side(reference_launch(lambda : source.recurrent_gated_delta_rule(*(arguments[name] for name in ('query', 'key', 'value', 'gate', 'beta')), initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=False)), (tolerance, tolerance), device_type='cuda', owners=locals())

def _gated_delta_inputs():
    (batch, length, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, length, heads, key_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, length, heads, value_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, length, heads), device='cuda', dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    return {'query': query, 'key': key, 'value': value, 'gate': gate, 'beta': beta, 'scale': key_dimension ** (-0.5)}
