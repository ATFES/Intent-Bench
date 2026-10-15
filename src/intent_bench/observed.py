"""Public outputs and in-place stores are both observable operator results."""


def with_writes(artifact):
    def run(*arguments):
        output = artifact.run(*arguments)
        interface = artifact.interface
        returned = () if not interface.outputs else (output,) if len(interface.outputs) == 1 else output
        values = {parameter.position: value for parameter, value in zip(interface.inputs, arguments, strict=True)
                  if parameter in interface.mutable_inputs}
        values.update((parameter.position, value) for parameter, value in zip(interface.outputs, returned, strict=True))
        observed = tuple(values[parameter.position] for parameter in interface.views if parameter.writable)
        return observed[0] if len(observed) == 1 else observed
    return run
