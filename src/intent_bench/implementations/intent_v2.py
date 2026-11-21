"""Convert a public V2 prepared call to the Bench operator protocol."""

from ..model import PreparedCall


def prepare(definition, args, env):
    import intentdsl

    target = env.options.get("compilerinfra", {})
    kernel = intentdsl.jit(
        definition,
        backend="triton",
        library=target.get("library"),
        configuration=target.get("configuration"),
    )
    call = kernel.prepare(*args)
    return PreparedCall(
        call.launch,
        call.result,
        compile=call.compile,
        device=env.device,
        metadata={
            "candidate": target.get("candidate"),
            "operator_entry": "prepared_launch",
        },
    )
