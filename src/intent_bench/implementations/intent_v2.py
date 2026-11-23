"""Convert a public V2 prepared call to the Bench operator protocol."""

from ..model import PreparedCall


def prepare(definition, args, env, backend="triton"):
    import intentdsl

    target = env.options.get("compilerinfra", {})
    kernel = intentdsl.jit(
        definition,
        backend=backend,
        library=target.get("library"),
        configuration=target.get("configuration"),
    )
    call = kernel.prepare(*args, artifact_file=target.get("artifact_file"))
    metadata={"candidate":target.get("candidate"),"operator_entry":"prepared_launch"}
    def compile():
        call.compile()
        if backend=="cutile":metadata["cold_compile"]=call.compilation_report()
    def result():
        if backend=="cutile":metadata["cold_compile"].update(call.compilation_report())
        return call.result()
    return PreparedCall(
        getattr(call, "execute", call.launch),
        result,
        compile=compile,
        device=env.device,
        tuning_report=call.tuning_report,
        metadata=metadata,
    )
