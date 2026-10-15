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
        for path in (root / "python", root / "examples"):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
    try:
        return importlib.import_module("intent")
    except ImportError as error:
        raise EnvironmentUnavailable("install intentdsl or supply --intent-root") from error


def load_workload(name):
    if name.startswith("advanced/"):
        from .workloads.advanced import build
        return build(name.split("/",1)[1])
    if name.startswith("rvv/"):
        from .workloads.rvv import build
        return build(name.split("/",1)[1])
    if name.startswith("mlu/"):
        from .workloads.mlu import build
        return build(name.split("/",1)[1])
    if name.startswith("cpu/"):
        from .workloads.cpu import build
        return build(name.split("/",1)[1])
    if name in {"softmax", "relu"}:
        from .workloads.elementwise import build
        return build(name)
    if name.startswith("corpus/"):
        from .catalog import get_workload
        return get_workload(name)
    value = load(name)
    return value() if callable(value) else value
