from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.gated_delta import recurrent_gated_delta_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.bangc_sequence import LocalSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    arguments = _gated_delta_inputs()
    sequence = LocalSequence(context)
    outputs = sequence.add(recurrent_gated_delta_fwd, arguments, constexprs={'HEAD_GROUP': 1})
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return checked_call(prepare_sequence(sequence, (outputs['output'], outputs['final_state'])), (tolerance, tolerance), device_type='mlu', owners=locals())

def _gated_delta_inputs():
    (batch, length, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, length, heads, key_dimension), device='cpu', dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, length, heads, value_dimension), device='cpu', dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, length, heads), device='cpu', dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    return {'query': query, 'key': key, 'value': value, 'gate': gate, 'beta': beta, 'scale': key_dimension ** (-0.5)}
