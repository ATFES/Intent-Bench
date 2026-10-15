"""Public-program checks use the author's entry and report their actual stage."""
from importlib import import_module
from pathlib import Path
import time
import traceback

from .loading import intent_environment, load
from .model import Environment, Result
from .results import save, table


def check_programs(args):
    env=Environment(intent_root=args.intent_root,compiler=args.compiler)
    intent_environment(env)
    entries=load(str(args.intent_root/"examples/use.py")+":EXAMPLES")
    from intent.tools.backends import make_target
    from programs.common import ExampleContext
    from programs import inputs
    names=args.program or list(entries)
    unknown=set(names)-entries.keys()
    if unknown:raise ValueError(f"unknown mainline programs: {unknown}")
    rows=[]
    device=getattr(args,'device',None) or 'auto'
    if device=='auto':device='cuda:0' if args.target in ('triton','cutile') else 'cpu'
    target_options={'device':int(device.partition(':')[2] or 0)} if args.target in ('triton','cutile') else {'workers':getattr(args,'threads',1)} if args.target in ('mojo','weft') else {}
    for name in names:
        context=None
        row=Result("mainline/"+name,"mainline","author",{},"intent:"+args.target,
                   device=device,stage=args.stage)
        start=time.perf_counter()
        try:
            inputs.seed([0,list(entries).index(name)])
            context=ExampleContext(make_target(args.target,target_options),row.device,compiler=args.compiler,
                                   prepared=True,stage=args.stage)
            module,_=entries[name]
            getattr(import_module("programs."+module),name)(context)
            row.status={"source":"generated","native":"compiled","run":"executed"}[args.stage]
            row.metadata["artifacts"]=[str(artifact.cache_directory) for artifact in context.artifacts]
            row.metadata["note"]="Author entry check; execution alone does not establish numerical correctness."
        except Exception as error:
            row.status="compilation_failed" if hasattr(error,"stage") else "execution_failed"
            row.stage=getattr(error,"stage",args.stage)
            row.error=str(error);row.traceback=traceback.format_exc()
        finally:
            row.stages_seconds[args.stage]=time.perf_counter()-start
            if context:context.close()
        rows.append(row);print(table([row]),flush=True)
    directory=save(rows,args.output,command=["intent-bench","check","--target",args.target,"--stage",args.stage])
    print(f"Results: {directory.resolve()}")
    return int(any(r.status.endswith("failed") for r in rows))
