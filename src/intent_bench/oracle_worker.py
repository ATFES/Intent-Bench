"""Run only registered correctness checks on transported actual inputs/results."""
import json
from pathlib import Path
import sys
from .oracle import validate_captured
from .output import run_output
from .request import RunRequest


def main():
    root = Path(sys.argv[1])
    request = RunRequest.read(json.loads((root/"request.json").read_text()))
    participants = json.loads((root/"participants.json").read_text())
    with run_output(request.environment, request.workload, root/"validation") as output:
        rows = validate_captured(request, root, participants)
        (output/"results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
