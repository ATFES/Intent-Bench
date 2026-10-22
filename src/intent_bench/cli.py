"""Human entrypoints over the same request service used by Python and MCP."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

def _assignment(text):
    key, separator, value = text.partition("=")
    if not key or not separator:
        raise ValueError("expected NAME=JSON")
    return key, json.loads(value)

def main(argv=None):
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments[:2] == ["experiment","agent"]:
        from .agent import main as agent_main
        return agent_main(arguments[2:])
    if arguments[:2] == ["experiment","paper"]:
        from .experiment import paper
        return paper(arguments[2:])
    parser = argparse.ArgumentParser(prog="intent-bench")
    commands = parser.add_subparsers(dest="command",required=True)
    listing = commands.add_parser("list",help="Discover registered workloads, programs and environments")
    listing.add_argument("--kind",choices=("workloads","implementations","programs","suites","environments"),default="workloads")
    listing.add_argument("--json",action="store_true")
    describe = commands.add_parser("describe")
    describe.add_argument("workload")
    environment = commands.add_parser("env")
    environment.add_argument("action",choices=("list","show"))
    environment.add_argument("name",nargs="?")
    plugin = commands.add_parser("plugin")
    plugin.add_argument("--output",type=Path,required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("before")
    comparison.add_argument("after")
    comparison.add_argument("--max-regression",type=float)
    report = commands.add_parser("report")
    report.add_argument("run")
    report.add_argument("--output",type=Path)
    experiment = commands.add_parser("experiment",help="Explicit paper and Agent protocols")
    experiment.add_argument("protocol",choices=("agent","paper"))
    run = commands.add_parser("run")
    run.add_argument("workload",nargs="?")
    run.add_argument("--suite",choices=("gpu","cpu","rvv","mlu","all"))
    run.add_argument("--impl",action="append")
    run.add_argument("--baseline")
    run.add_argument("--case",action="append")
    run.add_argument("--param",action="append",default=[])
    run.add_argument("--check-only",action="store_true")
    run.add_argument("--jobs",type=int,default=1)
    run.add_argument("--samples",type=int)
    run.add_argument("--warmup",type=int)
    run.add_argument("--cache",choices=("flush","warm"),default="flush")
    run.add_argument("--throughput",choices=("tflops","gbps"))
    run.add_argument("--timing",choices=("auto","host","cuda_event","cuda_graph","native"),default="auto")
    run.add_argument("--scope",choices=("operator","host_call"),default="operator")
    check = commands.add_parser("check")
    check.add_argument("--suite",choices=("mainline",),default="mainline")
    check.add_argument("--backend",required=True,choices=("triton","cutile","mojo","weft","bangc"))
    check.add_argument("--program",action="append")
    check.add_argument("--stage",choices=("source","native","run"),default="run")
    for command in (run,check):
        command.add_argument("--timeout",type=float,default=300)
        command.add_argument("--env")
        command.add_argument("--device")
        command.add_argument("--threads",type=int)
        command.add_argument("--seed",type=int,default=0)
        command.add_argument("--compiler")
        command.add_argument("--intent-root",type=Path,help="Verify an installed editable checkout; never modifies imports")
        command.add_argument("--option",action="append",default=[])
        command.add_argument("--output",type=Path)
        command.add_argument("--output-root",type=Path)
    args = parser.parse_args(arguments)
    from . import results
    if args.command == "plugin":
        from .plugins import export_plugin
        print(json.dumps(export_plugin(args.output),indent=2))
        return 0
    if args.command == "describe":
        from .discovery import describe_workload
        print(json.dumps(describe_workload(args.workload),indent=2))
        return 0
    if args.command == "env":
        from .environments import list_environments,load_environment,describe_environment
        if args.action == "show" and not args.name:
            parser.error("env show requires a name")
        value = list_environments() if args.action == "list" else describe_environment(load_environment(args.name))
        print(json.dumps(value,indent=2))
        return 0
    if args.command == "list":
        from .discovery import list_workloads,describe_implementation
        if args.kind == "workloads":
            value = list_workloads()
        elif args.kind == "implementations":
            value = [describe_implementation(name) for name in sorted({name for row in list_workloads() for name in row["implementations"]})]
        elif args.kind == "programs":
            from intent.examples import PROGRAMS
            value = [{"name":name,"description":description} for name,(_,description) in PROGRAMS.items()]
        elif args.kind == "environments":
            from .environments import list_environments
            value = list_environments()
        else:
            from .suites import SUITES
            value = [{"name":name,"providers":sorted(providers)} for name,providers in SUITES.items()]
        print(json.dumps(value,indent=2) if args.json else "\n".join(str(row) for row in value))
        return 0
    if args.command == "compare":
        changes = results.compare(results.read(args.before),results.read(args.after))
        print(results.comparison_table(changes))
        return int(args.max_regression is not None and any(r["speedup"] is not None and 1/r["speedup"]>1+args.max_regression for r in changes))
    if args.command == "report":
        path = Path(args.run)
        output = args.output or (path/"report.html" if path.is_dir() else path.with_suffix(".html"))
        results.write_html(results.read(path),output,source=path/"results.json" if path.is_dir() else path)
        print(output.resolve())
        return 0
    from .environments import load_environment
    env = load_environment(args.env,device=args.device,threads=args.threads,compiler=args.compiler,
                           intent_root=args.intent_root,output_root=args.output_root,
                           options=dict(_assignment(item) for item in args.option))
    from .request import RunRequest
    from .service import run as execute
    from .model import Measurement
    if args.command == "check":
        from intent.examples import PROGRAMS
        names = args.program or tuple(PROGRAMS)
        if set(names)-PROGRAMS.keys():
            parser.error("unknown program")
        code = 0
        for name in names:
            output = args.output/name if args.output else None
            outcome = execute(RunRequest(f"mainline/{args.backend}/{name}",("intent:"+args.backend,),
                environment=env,measurement=Measurement(enabled=False),output=output,seed=args.seed,
                kind="program",stage=args.stage,timeout=args.timeout))
            print(results.table(outcome.results),flush=True)
            print(f"Results: {outcome.directory}")
            code |= outcome.exit_code
        return code
    selected = tuple(name for group in args.impl or () for name in group.split(","))
    settings = Measurement(method=args.timing,samples=args.samples,warmup=args.warmup,scope=args.scope,
                           enabled=not args.check_only,cache=args.cache,throughput=args.throughput)
    if args.suite:
        if args.workload:
            parser.error("choose a workload or a suite")
        if args.jobs != 1 and not args.check_only:
            parser.error("performance runs are serial")
        from .suites import run_suite
        return run_suite(args,env,settings,selected)
    if not args.workload or not selected:
        parser.error("run requires a workload and explicit --impl")
    outcome = execute(RunRequest(args.workload,selected,env,settings,args.baseline,
        tuple(args.case or ()),dict(_assignment(item) for item in args.param),args.seed,args.output,timeout=args.timeout))
    print(results.table(outcome.results))
    print(f"Results: {outcome.directory}")
    return outcome.exit_code

if __name__ == "__main__":
    raise SystemExit(main())
