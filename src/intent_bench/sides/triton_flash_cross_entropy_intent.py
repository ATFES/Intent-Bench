from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.loss.cross_entropy import flash_cross_entropy_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (tokens, vocabulary) = (8192, 32768)
    logits = torch.randn((tokens, vocabulary), device='cuda', dtype=torch.bfloat16)
    labels = torch.randint(0, vocabulary, (tokens,), device='cuda', dtype=torch.int64)
    (_, generated) = compile_single(context, flash_cross_entropy_bf16, (logits, labels), constexprs={'IGNORE_INDEX': -100, 'Z_LOSS_SCALE': 0.0001})
    return checked_call(generated, (Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.05, rtol=0.01)), cuda_graph=False, owners=locals())
