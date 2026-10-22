import importlib
import importlib.util
from pathlib import Path
import sys
import uuid

from .model import EnvironmentUnavailable


def load(reference):
    module, separator, symbol = reference.rpartition(":")
    if not separator:
        module, symbol = reference, "bench"
    path = Path(module)
    if path.is_file():
        name = "_intent_bench_user_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, path.resolve())
        obj = importlib.util.module_from_spec(spec)
        sys.modules[name] = obj
        spec.loader.exec_module(obj)
    else:
        obj = importlib.import_module(module)
    return getattr(obj, symbol)


def intent_environment(environment):
    if environment.intent_root is not None:
        root = Path(environment.intent_root).resolve()
        if not (root / "python/intent").is_dir():
            raise EnvironmentUnavailable(f"Intent checkout not found: {root}")
    try:
        module = importlib.import_module("intent")
        if environment.intent_root is not None:
            expected = root / "python/intent"
            if Path(module.__file__).resolve().parent != expected:
                raise EnvironmentUnavailable("install this Intent checkout with pip install -e; --intent-root does not modify sys.path")
        return module
    except ImportError as error:
        raise EnvironmentUnavailable("install intentdsl or supply --intent-root") from error


def load_workload(name):
    from .registry import workloads
    registration = workloads().get(name)
    if registration is not None:
        return registration.build()
    value = load(name)
    return value() if callable(value) else value
