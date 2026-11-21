"""One configured output root for results, workers and compiler caches."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil


def output_root(explicit=None):
    selected = explicit or os.environ.get("INTENT_BENCH_OUTPUT_ROOT")
    if selected is None:
        config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home()/".config"))/"intent-bench/config.json"
        if config.is_file():
            selected = json.loads(config.read_text()).get("output_root")
    if not selected:
        raise ValueError("configure intent-bench output_root or set INTENT_BENCH_OUTPUT_ROOT")
    # Keep the configured spelling through symlinks. Expanding a storage mount
    # can exceed AF_UNIX's path limit in provider multiprocessing workers.
    return Path(selected).expanduser().absolute()


def component(value):
    # Human-readable paths, never fingerprints. Workload slashes remain hierarchy.
    return "/".join(re.sub(r"[^a-zA-Z0-9_.-]+", "_", p) or "item"
                    for p in str(value).split("/") if p not in {"", ".", ".."})


def destination(environment, workload, explicit=None):
    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    return output_root(environment.output_root)/"latest"/component(environment.name)/component(workload)


def worker_root():
    selected = os.environ.get("INTENT_BENCH_RUN_DIR")
    if not selected:
        raise RuntimeError("worker preparation requires the Bench run service")
    return output_root()/"work"


@contextmanager
def run_output(environment, workload, explicit=None, *, execution=True):
    root = output_root(environment.output_root)
    directory = destination(environment, workload, explicit)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent/("."+directory.name+".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another Bench run owns {directory}") from None
        # A failed or interrupted run must never expose the previous success.
        for filename in ("results.json", "results.csv", "report.html", "command.json", "worker.log"):
            (directory/filename).unlink(missing_ok=True)
        # Only invocation-owned diagnostics live here; compiled libraries and shared caches do not.
        if (directory/"artifacts").exists():
            shutil.rmtree(directory/"artifacts")
        if not execution:
            yield directory
            return
        cache = root/"cache"/component(environment.name)
        if environment.options.get("fresh_compile_cache"):
            import tempfile
            cache.mkdir(parents=True, exist_ok=True)
            cache = Path(tempfile.mkdtemp(dir=cache, prefix="cold-"))
        variables = {
            "INTENT_BENCH_OUTPUT_ROOT": str(root), "INTENT_BENCH_RUN_DIR": str(directory),
            "INTENT_CACHE_DIR": str(cache/"intent"), "XDG_CACHE_HOME": str(cache/"xdg"),
            "TRITON_CACHE_DIR": str(cache/"triton"), "CUDA_CACHE_PATH": str(cache/"cuda"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache/"inductor"),
            "TORCH_COMPILE_DEBUG_DIR": str(root/"work"/"inductor-debug"),
            "TMPDIR": str(root/"work"/"tmp"), "PYTHONDONTWRITEBYTECODE": "1",
        }
        Path(variables["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        for option, names in (("mojo", ("INTENT_MOJO", "INTENT_BENCH_MOJO")),
                              ("weft_compiler", ("INTENT_WEFT_COMPILER", "WEFT_COMPILER")),
                              ("triton_ptxas", ("TRITON_PTXAS_PATH",)),
                              ("triton_ptxas_blackwell", ("TRITON_PTXAS_BLACKWELL_PATH",)),
                              ("cncc", ("INTENT_BANGC_CNCC",))):
            if option in environment.options:
                for name in names:
                    variables[name] = str(environment.options[option])
        previous = {key: os.environ.get(key) for key in variables}
        os.environ.update(variables)
        import tempfile
        prior_tempdir, tempfile.tempdir = tempfile.tempdir, variables["TMPDIR"]
        try:
            yield directory
        finally:
            tempfile.tempdir = prior_tempdir
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
