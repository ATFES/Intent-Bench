from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.ragged.jagged_mean import jagged_mean
from intent_bench.corpus._common.model import Tolerance
from intent_bench.implementations.bangc_sequence import RemoteSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    (batch, features, max_length) = (512, 128, 128)
    lengths = torch.arange(batch, device='cuda', dtype=torch.int32) % max_length + 1
    offsets = torch.empty(batch + 1, device='cuda', dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn(int(offsets[-1].item()), features, device='cuda', dtype=torch.float32)
    sequence = RemoteSequence(context)
    output = sequence.add(jagged_mean, {'values': values, 'offsets': offsets})['output']
    return Side(prepare_sequence(sequence, output), Tolerance(atol=2e-05, rtol=1e-05), device_type='mlu', owners=locals())
