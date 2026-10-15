from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.loss.fused_linear_cross_entropy import linear_logits_chunk, cross_entropy_probability_chunk, cross_entropy_mean
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
TOKENS = 2048
HIDDEN = 4096
VOCABULARY = 32768
CHUNK_SIZE = 1024

def prepare(context):
    hidden = torch.randn((TOKENS, HIDDEN), dtype=torch.bfloat16)
    weight = torch.randn((VOCABULARY, HIDDEN), dtype=torch.bfloat16)
    target = torch.randint(0, VOCABULARY, (TOKENS,), dtype=torch.int64)
    logits = torch.empty((CHUNK_SIZE, VOCABULARY), dtype=torch.bfloat16)
    loss = torch.empty((TOKENS,), dtype=torch.float32)
    mean = torch.empty((1,), dtype=torch.float32)
    linear = intent.compile(linear_logits_chunk, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    probability = intent.compile(cross_entropy_probability_chunk, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    reduction = intent.compile(cross_entropy_mean, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def launch():
        for begin in range(0, TOKENS, CHUNK_SIZE):
            end = min(begin + CHUNK_SIZE, TOKENS)
            chunk = logits[:end - begin]
            linear(hidden[begin:end], weight, chunk)
            probability(chunk, target[begin:end], loss[begin:end])
        reduction(loss, mean)
    return Side(PreparedLaunch(launch, lambda : (loss, mean, logits)), (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.02, rtol=0.02)), cuda_graph=False, device_type='cpu', note='Three-kernel linear/cross-entropy/mean pipeline; reference preserves bf16 logits/probability scratch and f32 loss, including the final scratch output.', owners=locals())
