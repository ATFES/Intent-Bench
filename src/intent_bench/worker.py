"""Execute a serialized public request in the selected installed environment."""
import json
from pathlib import Path
import sys
from .request import RunRequest
from .service import execute_local

def main():
    request = RunRequest.read(json.loads(Path(sys.argv[1]).read_text()))
    return execute_local(request).exit_code

if __name__ == "__main__":
    raise SystemExit(main())
