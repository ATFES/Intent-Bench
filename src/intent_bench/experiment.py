"""Explicit experiment protocols, separate from daily measurement defaults."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from .output import component, output_root
from .request import RunRequest
from .service import run
from .results import save


def paper(argv):
    parser = argparse.ArgumentParser(prog="intent-bench experiment paper")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="JSON with a name and complete RunRequest records; measurement settings are explicit")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    name = component(protocol["name"])
    requests = []
    for record in protocol["requests"]:
        # A paper protocol cannot silently inherit daily measurement defaults.
        measurement = record.get("measurement", {})
        if not {"method", "scope", "cache", "samples", "warmup"}.issubset(measurement):
            raise ValueError("paper requests must declare method, scope, cache, samples and warmup")
        if measurement["samples"] is None or measurement["warmup"] is None:
            raise ValueError("paper sampling counts must be explicit")
        requests.append(RunRequest.read(record))
    root = output_root(args.output_root)/"experiments"/name/datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, code = [], 0
    for index, request in enumerate(requests):
        env = replace(request.environment, output_root=output_root(args.output_root))
        outcome = run(replace(request, environment=env,
                              output=root/f"{index+1:03d}"/component(env.name)/component(request.workload)))
        rows.extend(outcome.results)
        code |= outcome.exit_code
        save(rows, root, command={"protocol":str(args.protocol.resolve())})
    print(root)
    return code
