from intent.examples.kernels.streaming.mamba import mamba_chunk_state_bf16_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(mamba_chunk_state_bf16_fwd, constexprs={'HEAD_GROUP': 4})
    return checked_call(prepare_program(context, program), Tolerance(atol=0.02, rtol=0.02),
                        device_type='mlu', owners={'program': program})
