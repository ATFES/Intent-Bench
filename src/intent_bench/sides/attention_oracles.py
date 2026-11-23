"""Numerical judges for the materialized mixed-attention inputs."""
from intent_bench.corpus.cpu.baselines.pytorch.cpu_runtime import (
    grouped_flash_decode, prepare_attention_backward,
)
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall


def grouped(context):
    q, k, v, scale = context.inputs
    result = {}
    def launch():
        result["y"] = grouped_flash_decode(q, k, v, scale)
    return PreparedCall(launch, lambda: result["y"], device=str(q.device),
                        tolerance=Tolerance(atol=0.1, rtol=0.05))


def backward(context):
    q, k, v, _, dy, _, scale = context.inputs
    _, _, gradients = prepare_attention_backward(q, k, v, dy, scale)
    result = {}
    def launch():
        result["grads"] = gradients()
    return PreparedCall(launch, lambda: result["grads"], device=str(q.device),
                        tolerance=tuple(Tolerance(atol=t) for t in (0.125, 0.25, 0.125)))
