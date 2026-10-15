from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.loss.cross_entropy import fused_cross_entropy_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (tokens, vocabulary) = (8192, 32768)
    logits = torch.randn((tokens, vocabulary), device='cuda', dtype=torch.bfloat16)
    labels = torch.randint(0, vocabulary, (tokens,), device='cuda', dtype=torch.int64)
    labels[::17] = -100
    generated_logits = logits.clone()
    (_, generated_base) = compile_single(context, fused_cross_entropy_bf16, (generated_logits, labels), constexprs={'IGNORE_INDEX': -100})
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : (*generated_base.outputs(), generated_logits), prepare=lambda : generated_logits.copy_(logits))
    return Side(generated, (Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.0), Tolerance(atol=0.05, rtol=0.05)), cuda_graph=False, owners=locals())
