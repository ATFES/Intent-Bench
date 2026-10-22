from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.quantization.nvfp4 import nvfp4_quantize as nvfp4_quantize_definition
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call

def prepare(context):
    (rows, columns) = (8192, 4096)
    x = torch.randn((rows, columns), dtype=torch.bfloat16)
    global_scale = torch.ones((1,), dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8)
    (row_blocks, column_blocks) = (rows // 128, columns // 64)
    scales_shape = (row_blocks, column_blocks, 32, 4, 4)
    generated_packed = packed.clone()
    generated_scales = torch.empty(scales_shape, dtype=torch.uint8)
    artifact = context.compile(nvfp4_quantize_definition)

    def generated_outputs():
        return (generated_packed, generated_scales.view(-1))
    generated = intent_call(artifact, (x, global_scale, generated_packed, generated_scales),
                            observe=lambda _: generated_outputs())
    return checked_call(generated, Tolerance(atol=0.0), device_type='cpu', note='Prepared NVFP4 group16; original N8192/K4096, packing, scale and output ABI. Storage allocation outside timing.', owners=locals())
