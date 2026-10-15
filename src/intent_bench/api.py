from .model import Implementation, PreparedCall
from .trees import clone, restore


def callable_implementation(name, function, *, mutable=False, device="cpu", synchronize=None):
    """Wrap a normal callable. Inputs are positional tuples or keyword dictionaries."""
    def prepare(case, inputs, environment):
        initial = clone(inputs) if mutable else None
        state = {}
        def run():
            state["output"] = function(**inputs) if isinstance(inputs, dict) else function(*inputs)
        return PreparedCall(run, lambda: state["output"],
            reset=(lambda: restore(inputs, initial)) if mutable else None,
            device=device, synchronize=synchronize,scopes=("prepared_call","host_call"))
    return Implementation(name, prepare)
