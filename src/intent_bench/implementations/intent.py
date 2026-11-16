"""Bench configuration mapped to the ordinary product JIT, without hardware facts."""
from ..loading import intent_environment


def backend_options(environment, backend):
    provider_options = {}
    names = {"mojo": {"mojo": "compiler"},
             "weft": {"weft_options": "options"},
             "bangc": {"neuware": "neuware", "cncc": "compiler"}}.get(backend, {})
    for field, argument in names.items():
        if field in environment.options:
            provider_options[argument] = environment.options[field]
    return provider_options


def make_kernel(definition, environment, backend, *, constexprs=None, tuning_config=None, options=None):
    intent = intent_environment(environment)
    device = None if environment.device == "auto" else environment.device
    return intent.jit(definition, backend=backend, device=device, compiler=environment.compiler,
                      constexprs=constexprs, tuning_config=tuning_config, options=options,
                      workers=environment.threads if backend in {"mojo", "weft"} else None,
                      backend_options=backend_options(environment, backend))
