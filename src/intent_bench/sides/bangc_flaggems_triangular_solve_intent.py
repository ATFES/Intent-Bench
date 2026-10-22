from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.factorization.triangular_solve import batched_lower_triangular_solve
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.bangc_sequence import LocalSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    (batch, size) = (4096, 16)
    lower = torch.tril(torch.randn((batch, size, size), device='cpu', dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((batch, size), device='cpu', dtype=torch.float32)
    solution = initial.clone()
    sequence = LocalSequence(context)
    solution = sequence.add(batched_lower_triangular_solve, {'lower': lower, 'solution': solution})['solution']
    return checked_call(prepare_sequence(sequence, solution), Tolerance(atol=0.0001, rtol=0.0001), device_type='mlu', owners=locals())
