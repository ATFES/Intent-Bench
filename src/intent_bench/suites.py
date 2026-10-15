"""Suites select cases; subprocess isolation contains compiler/device failures."""
import os
from pathlib import Path
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed

from .catalog import catalog
from .model import Result
from .results import read,save


def run_suite(args):
    providers={"gpu":{"triton","cutile"},"cpu":{"mojo"},"rvv":{"weft"},"mlu":{"bangc"},
               "all":{"triton","cutile","mojo","weft","bangc"}}[args.suite]
    selected=[r for r in catalog() if r["provider"] in providers]
    root=args.output or Path("runs")/args.suite
    outcomes={}
    def execute(index,item):
        output=root/"cases"/item["provider"]/item["kernel"]
        command=[sys.executable,"-m","intent_bench.cli","run",item["name"],"--output",str(output),
                 "--threads",str(args.threads),"--cache",args.cache,
                 "--seed",str(args.seed),"--device",args.device,"--timing",args.timing,"--scope",args.scope]
        for flag,value in (("--intent-root",args.intent_root),("--compiler",args.compiler),("--baseline",args.baseline),
                           ("--samples",args.samples),("--warmup",args.warmup),("--throughput",args.throughput)):
            if value is not None:command.extend([flag,str(value)])
        if args.check_only:command.append('--check-only')
        for flag,values in (("--impl",args.impl or ["intent:"+item["provider"]]),("--option",args.option)):
            for value in values:command.extend([flag,value])
        output.mkdir(parents=True,exist_ok=True)
        if args.resume and (output/'results.json').is_file():
            return index,read(output/'results.json'),'existing result'
        # A failed new worker must never inherit the previous run's success.
        (output/'results.json').unlink(missing_ok=True)
        with (output/"worker.log").open("w") as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
        path=output/"results.json"
        if path.exists() and code!=124:
            measured=read(path)
            for row in measured:
                row.setdefault('metadata', {}).update(command=command, worker_log=str(output/'worker.log'))
            statuses=",".join(r["status"] for r in measured)
        else:
            names=[name for group in (args.impl or ['intent:'+item['provider']]) for name in group.split(',')]
            measured=[Result(item["name"],item["family"],item["case"],{},name,
                       status="timeout" if code==124 else "worker_failed",stage="worker",
                       metadata={'command':command},error=f"worker exit {code}; log: {output/'worker.log'}") for name in names]
            statuses=measured[0].status
        return index,measured,statuses
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        futures=[workers.submit(execute,index,item) for index,item in enumerate(selected)]
        for future in as_completed(futures):
            index,measured,statuses=future.result();outcomes[index]=measured
            rows=[row for i in sorted(outcomes) for row in outcomes[i]]
            save(rows,root,command=["intent-bench","run","--suite",args.suite])
            print(f"{len(outcomes)}/{len(selected)} {selected[index]['name']}: {statuses}",flush=True)
    rows=[row for i in sorted(outcomes) for row in outcomes[i]]
    return int(any((r.status if isinstance(r,Result) else r["status"]) not in {"passed","unchecked"} for r in rows))
