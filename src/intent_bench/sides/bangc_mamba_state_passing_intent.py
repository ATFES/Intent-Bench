from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.mamba import mamba_state_passing_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.bangc_sequence import LocalSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    states = torch.randn((1, 8, 32, 8192), device='cpu', dtype=torch.float32) * 0.01
    decay = -torch.rand((1, 32, 8), device='cpu', dtype=torch.float32) * 0.1
    initial = torch.zeros((1, 32, 8192), device='cpu', dtype=torch.float32)
    sequence = LocalSequence(context)
    outputs = sequence.add(mamba_state_passing_fwd, {'chunk_states': states, 'chunk_decay': decay, 'initial_states': initial})
    return checked_call(prepare_sequence(sequence, (outputs['states_before_chunk'], outputs['final_states'])), Tolerance(atol=1e-05, rtol=1e-05), device_type='mlu', owners=locals())
