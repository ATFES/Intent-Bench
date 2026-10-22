"""Background requests use the same service and result locations as foreground calls."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from .output import output_root
from .request import RunRequest


def start(requests):
    requests = (requests,) if isinstance(requests, RunRequest) else tuple(requests)
    if not requests:
        raise ValueError("a background job requires at least one request")
    root = output_root(requests[0].environment.output_root)/"work"/"jobs"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=root))
    (directory/"request.json").write_text(json.dumps([r.record() for r in requests], indent=2))
    with (directory/"worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "intent_bench.jobs", str(directory)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (directory/"job.json").write_text(json.dumps({"pid":process.pid, "state":"running"}))
    return {"run":str(directory), "state":"running", "log":str(directory/"worker.log")}


def status(directory):
    root = Path(directory).expanduser()
    completion = root/"completion.json"
    value = json.loads((completion if completion.exists() else root/"job.json").read_text())
    value.update(run=str(root), log=str(root/"worker.log"))
    if completion.exists():
        from .results import read
        value["results"] = [row for path in value.get("outputs", []) for row in read(path)]
    return value


def main(directory):
    from .service import run
    root = Path(directory)
    requests = [RunRequest.read(value) for value in json.loads((root/"request.json").read_text())]
    code, outputs = 0, []
    try:
        for request in requests:
            outcome = run(request)
            code |= outcome.exit_code
            outputs.append(str(outcome.directory))
    except Exception:
        traceback.print_exc()
        code = 1
    value = {"state":"completed", "exit_code":code, "outputs":outputs}
    temporary = root/"completion.tmp"
    temporary.write_text(json.dumps(value))
    temporary.replace(root/"completion.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
