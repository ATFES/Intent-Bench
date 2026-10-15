from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.attention_f32 import causal_attention_f32
from intent_bench.corpus._common.model import Tolerance
from intent_bench.implementations.bangc_sequence import RemoteSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    scale = 32 ** (-0.5)
    sequence = RemoteSequence(context)
    output = sequence.add(causal_attention_f32, {'q': q, 'k': k, 'v': v, 'scale': scale})['output']
    return Side(prepare_sequence(sequence, output), Tolerance(0.0002, 1e-05), device_type='mlu', owners=locals())
