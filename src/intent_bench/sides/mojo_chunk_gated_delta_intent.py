from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.gated_delta import chunk_gated_delta_prepare, chunk_gated_delta_recurrence
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    query = torch.randn((2, 2048, 8, 128), dtype=torch.bfloat16) * 0.1
    (key, value) = (torch.randn_like(query) * 0.1, torch.randn_like(query) * 0.1)
    gate = -torch.rand((2, 2048, 8), dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    arguments = (query, key, value, gate, beta, 128 ** (-0.5))
    (preparation, recurrence) = (context.compile(definition) for definition in (chunk_gated_delta_prepare, chunk_gated_delta_recurrence))
    (generated, source) = ({}, {})

    def launch():
        generated['output'] = recurrence.run(*preparation.run(*arguments))
    sequence = IntentSequence()
    intermediate = sequence.add(preparation, *arguments)
    output = sequence.add(recurrence, *intermediate)
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, (Tolerance(atol=0.1, rtol=0.05), Tolerance(atol=0.1, rtol=0.05)), device_type='cpu', note='Prepared B2/S2048/H8/K128/V128 BF16 chunk64 gated delta. Original prepare+recurrence, intermediate ABI, outputs and tolerances.', owners=locals())
