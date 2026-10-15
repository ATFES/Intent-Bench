from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.gated_delta import chunk_gated_delta_prepare, chunk_gated_delta_recurrence
from intent_bench.corpus._common.model import Tolerance
from intent_bench.implementations.bangc_sequence import RemoteSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    arguments = _gated_delta_inputs()
    sequence = RemoteSequence(context)
    intermediate = sequence.add(chunk_gated_delta_prepare, arguments)
    outputs = sequence.add(chunk_gated_delta_recurrence, intermediate)
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return Side(prepare_sequence(sequence, (outputs['output'], outputs['final_state'])), (tolerance, tolerance), device_type='mlu', owners=locals())

def _gated_delta_inputs():
    (batch, length, heads, key_dimension, value_dimension) = (2, 2048, 8, 128, 128)
    query = torch.randn((batch, length, heads, key_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, length, heads, value_dimension), device='cuda', dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, length, heads), device='cuda', dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    return {'query': query, 'key': key, 'value': value, 'gate': gate, 'beta': beta, 'scale': key_dimension ** (-0.5)}
