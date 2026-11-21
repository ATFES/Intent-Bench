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
            "artifact_file": candidate.get("artifact_file"),
        },
    }
    if "row_id" in selection:
        options["compilerinfra"]["row_id"] = selection["row_id"]
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
    result = {"directory": str(outcome.directory), "exit_code": outcome.exit_code}
    if candidate.get("artifact_file"):
        from .results import record
        result["measurements"] = []
        for raw in outcome.results:
            row = raw if isinstance(raw, dict) else record(raw)
            if not row["implementation"].startswith("intent-v2:"):
                continue
            actual = row["metadata"].get("selection", {})
            result["measurements"].append(dict(
                measurement_receipt=row["metadata"].get("measurement_receipt"),
                case=row["case"], status=row["status"], runtime_identity=actual.get("runtime_identity"),
                row_id=actual.get("winner_row_id"),
                artifact_sha256=actual.get("artifact_sha256"), measured_ns=(row["median_ms"] * 1e6
                if row["median_ms"] is not None else None)))
    return result
