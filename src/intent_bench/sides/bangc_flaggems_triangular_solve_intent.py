from intent.examples.kernels.factorization.triangular_solve import batched_lower_triangular_solve
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = context.compile(batched_lower_triangular_solve)
    call = prepare_program(context, program, mutable_positions=(1,),
                           result=lambda call, arguments: arguments[1])
    return checked_call(call, Tolerance(atol=0.0001, rtol=0.0001),
                        device_type='mlu', owners={'program': program})
