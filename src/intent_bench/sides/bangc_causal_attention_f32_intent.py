from intent.examples.kernels.streaming.attention_f32 import causal_attention_f32
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(causal_attention_f32)
    return checked_call(prepare_program(context, program), Tolerance(0.0002, 1e-05),
                        device_type='mlu', owners={'program': program})
