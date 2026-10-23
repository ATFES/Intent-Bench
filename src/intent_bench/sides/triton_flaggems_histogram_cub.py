import torch

from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.cub import histogram
from intent_bench.implementations.prepared import checked_call


def prepare(context):
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), device='cuda', dtype=torch.int32).to(torch.float32)
    call = histogram(samples, context.environment)
    return checked_call(call, Tolerance(atol=0.0), owners={"call_buffers": call.owners})
