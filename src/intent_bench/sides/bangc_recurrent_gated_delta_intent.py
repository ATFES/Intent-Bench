from intent.examples.kernels.streaming.gated_delta import recurrent_gated_delta_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(recurrent_gated_delta_fwd, constexprs={'HEAD_GROUP': 1})
    return checked_call(prepare_program(context, program), (Tolerance(atol=0.1, rtol=0.05), Tolerance(atol=0.1, rtol=0.05)),
                        device_type='mlu', owners={'program': program})
