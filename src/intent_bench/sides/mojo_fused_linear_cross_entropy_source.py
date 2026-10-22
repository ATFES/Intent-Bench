from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
TOKENS = 2048
HIDDEN = 4096
VOCABULARY = 32768
CHUNK_SIZE = 1024

def prepare(context):
    hidden = torch.randn((TOKENS, HIDDEN), dtype=torch.bfloat16)
    weight = torch.randn((VOCABULARY, HIDDEN), dtype=torch.bfloat16)
    target = torch.randint(0, VOCABULARY, (TOKENS,), dtype=torch.int64)
    source_state = {}

    def source_launch():
        source_state['outputs'] = reference(hidden, weight, target, CHUNK_SIZE)
    return checked_call(PreparedCall(source_launch, lambda : source_state['outputs']), (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.02, rtol=0.02)), cuda_graph=False, device_type='cpu', note='Three-kernel linear/cross-entropy/mean pipeline; reference preserves bf16 logits/probability scratch and f32 loss, including the final scratch output.', owners=locals())
