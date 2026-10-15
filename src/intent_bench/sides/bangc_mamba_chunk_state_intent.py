from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.mamba import mamba_chunk_state_bf16_fwd
from intent_bench.corpus._common.model import Tolerance
from intent_bench.implementations.bangc_sequence import RemoteSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    x = torch.randn((1, 2048, 32, 64), device='cuda', dtype=torch.bfloat16)
    basis = torch.randn((1, 2048, 8, 128), device='cuda', dtype=torch.bfloat16)
    dt = torch.rand((1, 32, 8, 256), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    sequence = RemoteSequence(context)
    output = sequence.add(mamba_chunk_state_bf16_fwd, {'state_basis': basis, 'x': x, 'dt': dt, 'cumulative_decay': decay}, constexprs={'HEAD_GROUP': 4})['states']
    return Side(prepare_sequence(sequence, output), Tolerance(atol=0.02, rtol=0.02), device_type='mlu', owners=locals())
