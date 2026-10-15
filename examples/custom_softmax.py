"""Add a mutable, multi-kernel implementation to an existing workload.

intent-bench run examples/custom_softmax.py:bench --case 64x1024-float16 \
    --impl mine,triton,cutile,intent:triton,intent:cutile --device cuda:0 \
    --intent-root /path/to/intentdsl
"""
from dataclasses import replace

from intent_bench import Implementation, PreparedCall
from intent_bench.workloads.elementwise import build


def prepare(case, inputs, environment):
    import torch

    device = "cuda:0" if environment.device == "auto" else environment.device
    original = torch.as_tensor(inputs[0], device=device)
    value = original.float().clone()
    initial = value.clone()
    output = {}

    def run():
        value.sub_(value.amax(dim=-1, keepdim=True))
        value.exp_()
        value.div_(value.sum(dim=-1, keepdim=True))
        output["value"] = value.to(original.dtype)

    return PreparedCall(run, lambda: output["value"],
                        reset=lambda: value.copy_(initial), device=device)


base = build("softmax")
bench = replace(base, implementations={**base.implementations,
                                      "mine": Implementation("mine", prepare)})
