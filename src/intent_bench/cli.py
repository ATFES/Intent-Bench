from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from .engine import benchmark
from .loading import load
from .model import Case, Environment, Measurement
from . import results


def workload(name):
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


def main(argv=None):
    arguments=list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0]=='agent':
        from .agent import main as agent_main
        return agent_main(arguments[1:])
    parser = argparse.ArgumentParser(prog="intent-bench", description="Compare operator implementations on shared workloads")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="Discover workloads without loading compiler or device SDKs")
    listing.add_argument("--json", action="store_true")
    run = sub.add_parser("run", help="Measure a built-in workload or file.py:bench")
    run.add_argument("workload",nargs="?")
    run.add_argument("--suite",choices=("gpu","cpu","rvv","mlu","all"))
    run.add_argument("--timeout",type=float,default=300,help="Per-workload subprocess timeout for a suite")
    run.add_argument("--check-only",action="store_true",help="Check compilation/execution/correctness without measuring performance")
    run.add_argument("--jobs",type=int,default=1,help="Concurrent suite checks; performance runs use one worker")
    run.add_argument("--resume",action="store_true",help="Explicitly reuse completed cases from this suite output directory")
    run.add_argument("--impl", action="append", help="Implementation names, comma separated; repeatable")
    run.add_argument("--baseline")
    run.add_argument("--case", action="append")
    run.add_argument("--param", action="append", default=[], metavar="NAME=JSON", help="Override a parameter; a JSON list creates a sweep")
    run.add_argument("--device", default="auto")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--samples", type=int, help="Override automatic sample count")
    run.add_argument("--warmup", type=int, help="Override automatic warmup count")
    run.add_argument("--cache", choices=("flush", "warm"), default="flush", help="CUDA L2 policy (default: flush before each sample)")
    run.add_argument("--throughput", choices=("tflops", "gbps"), help="Show one optional throughput metric when the workload defines it")
    run.add_argument("--timing", default="auto", choices=("auto","host","cuda_event","cuda_graph","native"))
    run.add_argument("--scope", default="prepared_call", choices=("prepared_call","host_call"))
    run.add_argument("--intent-root", type=Path)
    run.add_argument("--compiler")
    run.add_argument("--option", action="append", default=[], metavar="NAME=JSON")
    run.add_argument("--output", type=Path)
    comparison = sub.add_parser("compare", help="Compare two result runs")
    comparison.add_argument("before"); comparison.add_argument("after")
    comparison.add_argument("--max-regression", type=float, help="Optional fractional slowdown limit")
    report = sub.add_parser("report", help="Render a saved run")
    report.add_argument("run"); report.add_argument("--output", type=Path)
    check = sub.add_parser("check", help="Check mainline public programs")
    check.add_argument("--intent-root", type=Path, required=True)
    check.add_argument("--suite", choices=("mainline",), default="mainline")
    check.add_argument("--target", default="triton")
    check.add_argument("--program", action="append")
    check.add_argument("--stage", choices=("source","native","run"), default="run")
    check.add_argument("--compiler")
    check.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "list":
        entries = [{"name":name,"family":workload(name).family,"implementations":list(workload(name).implementations)}
                   for name in ("softmax","relu")]
        from .catalog import catalog,ADVANCED_OPERATIONS
        entries.extend(catalog())
        entries.extend({'name':'advanced/'+name,'family':'advanced','implementations':['torch','mojo','intent:mojo']} for name in ADVANCED_OPERATIONS)
        from .implementations.mojo import NAMES
        entries.extend({"name":"cpu/"+name,"family":"cpu","implementations":["numpy","mojo"]} for name in NAMES)
        from .workloads.mlu import NAMES as MLU_NAMES
        entries.extend({"name":"mlu/"+name,"family":"mlu","implementations":["cnnl","intent:bangc"]} for name in MLU_NAMES)
        from .workloads.rvv import NAMES as RVV_NAMES
        entries.extend({"name":"rvv/"+name,"family":"rvv","implementations":["ggml","intent:weft"]} for name in RVV_NAMES)
        print(json.dumps(entries, indent=2) if args.json else "\n".join(
            f"{x['name']}: {', '.join(x['implementations'])}" for x in entries))
        return 0
    if args.command == "compare":
        changes = results.compare(results.read(args.before),results.read(args.after))
        print(results.comparison_table(changes))
        return int(args.max_regression is not None and any(
            r["speedup"] is not None and 1/r["speedup"] > 1+args.max_regression for r in changes))
    if args.command == "report":
        source=Path(args.run)
        output = args.output or (source/"report.html" if source.is_dir() else source.with_suffix('.html'))
        results.write_html(results.read(args.run),output,source=source/'results.json' if source.is_dir() else source)
        print(output.resolve()); return 0
    if args.command == "check":
        from .mainline import check_programs
        return check_programs(args)
    if args.suite:
        if args.workload:parser.error("choose a workload or --suite")
        if args.jobs<1:parser.error('jobs must be positive')
        if args.jobs>1 and not args.check_only:parser.error('parallel suite workers require --check-only; performance runs are serialized')
        from .suites import run_suite
        return run_suite(args)
    if not args.workload:parser.error("run requires a workload or --suite")
    try:
        bench = workload(args.workload)
    except ImportError as error:
        from .model import Result
        from .catalog import catalog
        entry=next((row for row in catalog() if row['name']==args.workload),{})
        selected=[name for group in args.impl for name in group.split(',')] if args.impl else entry.get('implementations',['requested'])
        rows=[Result(args.workload,entry.get('family','custom'),case,{},name,status='environment_missing',stage='load',
                     device=args.device,error=f'{type(error).__name__}: {error}')
              for case in (args.case or [entry.get('case','load')]) for name in selected]
        print(results.table(rows))
        directory=results.save(rows,args.output,command=['intent-bench',*arguments])
        print(f'Results: {directory.resolve()}')
        return 1
    if args.case:
        known = {c.name for c in bench.cases}
        if set(args.case)-known:
            parser.error(f"unknown cases: {set(args.case)-known}")
        bench = replace(bench,cases=[c for c in bench.cases if c.name in args.case])
    if args.param:
        from itertools import product
        values = dict(_assignment(text) for text in args.param)
        if bench.parameter_keys is not None and set(values)-set(bench.parameter_keys):
            parser.error("this workload does not expose those parameters; use an explicit custom workload for a different input contract")
        keys = list(values)
        cases = []
        for base in bench.cases:
            for items in product(*(v if isinstance(v,list) else [v] for v in values.values())):
                parameters = {**base.parameters,**dict(zip(keys,items))}
                name = ",".join(f"{k}={v}" for k,v in parameters.items())
                if not any(c.name==name for c in cases):
                    cases.append(Case(name,parameters))
        bench = replace(bench,cases=cases)
    selected = [name for group in args.impl for name in group.split(",")] if args.impl else None
    if selected is not None:
        unknown=set(selected)-bench.implementations.keys()
        if unknown: parser.error(f"unknown implementations: {sorted(unknown)}")
    env=Environment(args.device,args.intent_root,args.compiler,args.threads,dict(_assignment(x) for x in args.option))
    settings=Measurement(args.timing,args.samples,args.warmup,args.scope,not args.check_only,
                         cache=args.cache,throughput=args.throughput)
    rows=benchmark(bench,selected,environment=env,measurement=settings,baseline=args.baseline,seed=args.seed)
    print(results.table(rows),flush=True)
    directory=results.save(rows,args.output,command=["intent-bench",*(argv or sys.argv[1:])])
    print(f"Results: {directory.resolve()}")
    return int(any(r.status not in {"passed","unchecked"} for r in rows))


def _assignment(text):
    key,sep,value=text.partition("=")
    if not key or not sep:
        raise ValueError("expected NAME=JSON")
    return key,json.loads(value)


if __name__ == "__main__":
    raise SystemExit(main())
