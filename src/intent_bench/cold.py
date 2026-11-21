"""Fresh process and compiler caches per participant and case."""

from contextlib import contextmanager
from dataclasses import replace
import os
import json
import re
from pathlib import Path

from .output import component, run_output
from .results import save, record


@contextmanager
def compiler_events(request):
    if not request.environment.options.get("fresh_compile_cache"):
        yield None
        return
    (impl,) = request.implementations
    report = {
        "cache_root": os.environ["INTENT_CACHE_DIR"],
        "rows": [],
        "candidate_count": None,
        "instrumentation": "unavailable",
    }
    if impl in {"torch", "ggml", "aten:cpu"}:
        report.update(
            candidate_count=0,
            instrumentation="prebuilt operator library; native measurement wrapper excluded",
        )
    if impl == "intent:mojo":
        yield report
        root = Path(report["cache_root"])
        for timing in sorted(
            (root / "mojo").glob("*/attempts/*/native_compilation.seconds")
        ):
            build = timing.parent
            source = Path(json.loads((build / "request.json").read_text())["source"])
            entry = re.search(r'@export\("([^"]+)"\)', source.read_text())
            report["rows"].append(
                {
                    "symbol": entry.group(1) if entry else source.parent.name,
                    "compiler_ms": 1000 * float(timing.read_text()),
                    "cache_hit": False,
                }
            )
        report.update(
            candidate_count=len(report["rows"]),
            instrumentation="Mojo native compiler command logs in empty per-case cache",
        )
        return
    if "triton" not in impl and impl != "intent:weft":
        yield report
        return
    from triton import knobs

    previous = knobs.compilation.listener

    def event(**data):
        source = data["src"]
        metadata = data["metadata"]
        report["rows"].append(
            {
                "symbol": str(getattr(source, "name", source)),
                "artifact": metadata.get("hash"),
                "constants": str(getattr(source, "constants", {})),
                "warps": metadata.get("num_warps"),
                "stages": metadata.get("num_stages"),
                "compiler_ms": data["times"].total / 1000,
                "cache_hit": bool(data["cache_hit"]),
            }
        )
        if previous:
            previous(**data)

    knobs.compilation.listener = event
    try:
        yield report
    finally:
        knobs.compilation.listener = previous
        keys = {row["artifact"] for row in report["rows"]}
        report.update(
            candidate_count=len(keys) if None not in keys else None,
            compile_events=len(report["rows"]),
            instrumentation="Triton compilation listener",
        )
        if impl == "intent:weft":
            report["host_builds"] = [
                {"compiler_ms": 1000 * float(t.read_text()), "cache_hit": False}
                for t in Path(report["cache_root"]).glob(
                    "weft-host/*/attempts/*/provider_host_compilation.seconds"
                )
            ]


def isolated(request, dispatch):
    from .service import RunOutcome
    from .discovery import describe_workload

    cases = request.cases or tuple(
        c["name"] for c in describe_workload(request.workload)["cases"]
    )
    env = replace(
        request.environment,
        options={**request.environment.options, "fresh_compile_cache": True},
    )
    collected = []
    with run_output(
        env, request.workload, request.output, execution=False
    ) as directory:
        for i, case in enumerate(cases):
            group = []
            for j, impl in enumerate(request.implementations):
                task = replace(
                    request,
                    environment=env,
                    implementations=(impl,),
                    cases=(case,),
                    baseline=impl,
                    output=directory
                    / "artifacts"
                    / str(i)
                    / (str(j) + "-" + component(impl)),
                )
                part = dispatch(task)
                group.extend(record(row) for row in part.results)
            ref = {
                r["case"]: r.get("median_ms")
                for r in group
                if r["implementation"] == request.baseline and r["status"] == "passed"
            }
            for r in group:
                base = ref.get(r["case"])
                r.update(
                    baseline=request.baseline,
                    baseline_ms=base,
                    speedup=(
                        base / r["median_ms"]
                        if base and r.get("median_ms") and r["status"] == "passed"
                        else None
                    ),
                )
            collected.extend(group)
        save(collected, directory, command=request.record())
        return RunOutcome(collected, directory)
