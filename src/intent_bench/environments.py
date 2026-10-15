"""Named execution settings, independent of compiler backend selection."""
from dataclasses import asdict, fields
import json
import os
from pathlib import Path

from .model import Environment


def environment_directory():
    configured = os.environ.get("INTENT_BENCH_ENVIRONMENTS")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home()/".config"))/"intent-bench/environments"


def environment_path(name):
    path = Path(name).expanduser()
    if path.is_file():
        return path.resolve()
    path = environment_directory()/(name+".json")
    if not path.is_file():
        raise FileNotFoundError(f"environment {name!r} not found in {environment_directory()}")
    return path


def load_environment(name=None, **overrides):
    values = json.loads(environment_path(name).read_text()) if name else {}
    values.pop("description", None)
    allowed = {field.name for field in fields(Environment)}
    unknown = set(values)-allowed
    if unknown:
        raise ValueError(f"unknown environment fields: {sorted(unknown)}")
    options = {**values.get("options", {}), **(overrides.pop("options", None) or {})}
    values.update({key:value for key,value in overrides.items() if value is not None})
    values["options"] = options
    values.setdefault("device", "auto")
    if values.get("intent_root"):
        values["intent_root"] = Path(values["intent_root"]).expanduser()
    if values.get("transport") not in (None,"local","ssh"):
        raise ValueError("environment transport must be local or ssh")
    host = values.get("host") or options.get("host")
    if values.get("transport") == "ssh" and not host:
        raise ValueError("SSH environment requires a host alias")
    if values.get("transport") == "local" and host:
        raise ValueError("local environment must not specify an SSH host")
    return Environment(**values)


def describe_environment(environment):
    values = asdict(environment)
    if values["intent_root"] is not None:
        values["intent_root"] = str(values["intent_root"])
    return values


def list_environments():
    return [{"name":path.stem,"path":str(path),**json.loads(path.read_text())}
            for path in sorted(environment_directory().glob("*.json"))]
