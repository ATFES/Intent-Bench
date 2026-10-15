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
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flash-attention/loss/cross_entropy/cross_entropy_runtime.py', 'intent_v2_triton_flash_cross_entropy')
    state: dict[str, object] = {}

    def source_launch():
        state['outputs'] = runtime.upstream((logits, labels))
    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda : state['outputs'])
    return Side(source, (Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.05, rtol=0.01)), cuda_graph=False, owners=locals())
