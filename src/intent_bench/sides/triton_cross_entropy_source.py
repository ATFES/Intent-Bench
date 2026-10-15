from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (tokens, vocabulary) = (8192, 32768)
    logits = torch.randn((tokens, vocabulary), device='cuda', dtype=torch.bfloat16)
    labels = torch.randint(0, vocabulary, (tokens,), device='cuda', dtype=torch.int64)
    labels[::17] = -100
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/liger-kernel/loss/cross_entropy/cross_entropy_runtime.py', 'intent_v2_triton_cross_entropy')
    source_logits = logits.clone()
    state: dict[str, object] = {}

    def source_launch():
        state['outputs'] = runtime.upstream((source_logits, labels))
    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda : state['outputs'], prepare=lambda : source_logits.copy_(logits))
    return Side(source, (Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.0), Tolerance(atol=0.05, rtol=0.05)), cuda_graph=False, owners=locals())
