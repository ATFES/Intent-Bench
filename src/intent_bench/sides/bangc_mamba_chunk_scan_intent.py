from intent.examples.kernels.streaming.selective_scan import mamba_chunk_scan_bf16_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(mamba_chunk_scan_bf16_fwd, constexprs={'HEADS_PER_GROUP': 4})
    return checked_call(prepare_program(context, program), Tolerance(atol=0.1, rtol=0.05),
                        device_type='mlu', owners={'program': program})
