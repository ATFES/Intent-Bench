from intent.examples.kernels.ragged.jagged_mean import jagged_mean
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(jagged_mean)
    return checked_call(prepare_program(context, program), Tolerance(atol=2e-05, rtol=1e-05),
                        device_type='mlu', owners={'program': program})
