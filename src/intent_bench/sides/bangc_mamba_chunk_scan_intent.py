from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.selective_scan import mamba_chunk_scan_bf16_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.bangc_sequence import LocalSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    x = torch.randn((1, 2048, 32, 64), device='cpu', dtype=torch.bfloat16)
    matrix = torch.randn((1, 2048, 8, 128), device='cpu', dtype=torch.bfloat16)
    cb = torch.randn((1, 8, 8, 256, 256), device='cpu', dtype=torch.bfloat16) * 0.01
    dt = torch.rand((1, 32, 8, 256), device='cpu', dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((1, 8, 32, 64, 128), device='cpu', dtype=torch.float32) * 0.01
    residual = torch.randn((32,), device='cpu', dtype=torch.float32) * 0.01
    sequence = LocalSequence(context)
    output = sequence.add(mamba_chunk_scan_bf16_fwd, {'cb': cb, 'x': x, 'dt': dt, 'dA_cumsum': decay, 'state_matrix': matrix, 'previous_states': previous, 'residual_scale': residual}, constexprs={'HEADS_PER_GROUP': 4})['output']
    return checked_call(prepare_sequence(sequence, output), Tolerance(atol=0.1, rtol=0.05), device_type='mlu', owners=locals())
