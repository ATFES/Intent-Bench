from intent.examples.kernels.streaming.mamba import mamba_state_passing_fwd
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(mamba_state_passing_fwd)
    return checked_call(prepare_program(context, program), Tolerance(atol=1e-05, rtol=1e-05),
                        device_type='mlu', owners={'program': program})
