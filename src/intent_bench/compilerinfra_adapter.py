"""Measurement service for a locally installed CompilerInfra candidate."""

from dataclasses import replace
from pathlib import Path
from .environments import load_environment
from .model import Measurement
from .request import RunRequest
from .service import run


def measure(candidate):
    selection = candidate["workload"]
    env = load_environment(selection.get("environment"))
    if env.transport == "ssh":
        raise ValueError(
            "open and measure the installed candidate on the execution host"
        )
    options = {
        **env.options,
        "compilerinfra": {
            "library": candidate["library"],
            "configuration": candidate["configuration"],
            "candidate": candidate["candidate"],
        },
    }
    env = replace(env, python=candidate["python"], options=options)
    request = RunRequest(
        selection["name"],
        tuple(selection["implementations"]),
        env,
        Measurement(**selection.get("measurement", {})),
        baseline=selection.get("baseline"),
        cases=tuple(selection["cases"]),
        output=Path(candidate["output_dir"]),
    )
    outcome = run(request)
    return {"directory": str(outcome.directory), "exit_code": outcome.exit_code}
