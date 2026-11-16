"""Bench configuration mapped to the ordinary product JIT, without hardware facts."""
from ..loading import intent_environment


def backend_options(environment, backend):
    return dict(environment.options.get("intent", {}).get(backend, {}))


def execution(environment, backend):
    """Select the configured device using the product declaration, without a probe."""
    intent_environment(environment)
    from intent.targets.provider import provider
    from intent.targets.local import select_device
    declared = provider(backend).local
    device = environment.device
    if device == "auto":
        device = declared.device_kind + (":0" if declared.indexed else "")
    selected = select_device(backend, (), device)
    return str(selected), environment.threads if selected.kind == "cpu" else None


def make_kernel(definition, environment, backend, *, constexprs=None, tuning_config=None, options=None):
    intent = intent_environment(environment)
    device = None if environment.device == "auto" else environment.device
    _, workers = execution(environment, backend)
    return intent.jit(definition, backend=backend, device=device, compiler=environment.compiler,
                      constexprs=constexprs, tuning_config=tuning_config, options=options,
                      workers=workers,
                      backend_options=backend_options(environment, backend))
