from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, IntentSequence
import intent
import torch
from intent.examples.kernels.loss.fused_linear_cross_entropy import linear_logits_chunk, cross_entropy_probability_chunk, cross_entropy_mean
from intent_bench.implementations.context import Tolerance
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
    linear = context.compile(linear_logits_chunk)
    probability = context.compile(cross_entropy_probability_chunk)
    reduction = context.compile(cross_entropy_mean)

    def launch():
        for begin in range(0, TOKENS, CHUNK_SIZE):
            end = min(begin + CHUNK_SIZE, TOKENS)
            chunk = logits[:end - begin]
            linear(hidden[begin:end], weight, chunk)
            probability(chunk, target[begin:end], loss[begin:end])
        reduction(loss, mean)
    sequence = IntentSequence()
    for begin in range(0, TOKENS, CHUNK_SIZE):
        end = min(begin + CHUNK_SIZE, TOKENS)
        chunk = logits[:end - begin]
        sequence.add(linear, hidden[begin:end], weight, outputs=(chunk,))
        sequence.add(probability, chunk, target[begin:end], outputs=(loss[begin:end],))
    sequence.add(reduction, loss, outputs=(mean,))
    generated = sequence.prepared(lambda: (loss, mean, logits), host_run=launch)
    return checked_call(generated, (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.02, rtol=0.02)), cuda_graph=False, device_type='cpu', note='Complete chunked linear/cross-entropy/mean native sequence; reference preserves bf16 logits/probability scratch and f32 loss, including the final scratch output.', owners=locals())
