"""Agent generation is a producer; candidate evaluation uses the ordinary engine."""
import argparse
from dataclasses import asdict
from importlib import import_module
import inspect
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

from .engine import benchmark
from .loading import intent_environment, load
from .model import Case, Environment, Implementation, Measurement, PreparedCall, Workload
from .results import save,table


def task_data(args):
    from .corpus.agent_tritonbench import tasks
    suite=tasks.read_suite(args.suite)
    task=next((x for x in suite['tasks'] if x['id']==args.task),None)
    if task is None:raise ValueError(f"task {args.task} is not in the selected suite")
    row=next(x for x in tasks.catalog(args.reference,suite) if x['task']==args.task)
    return tasks,suite,task,row


def evaluate(args):
    tasks,suite,task,row=task_data(args)
    reference=tasks.reference(args.reference,row)
    env=Environment(args.device,args.intent_root,args.compiler)
    def inputs(case,seed):
        local_suite={**suite,'input_seed':seed}
        return tasks.invocation(args.reference,row,task,local_suite,device='cpu')
    def prepare(function,call):
        call=call.to_device(args.device)
        from .trees import clone, restore
        initial=clone((call.args,call.kwargs))
        state={}
        def run():
            value=call.call(function)
            if call.bound.get('out') is not None and value is not call.bound['out']:
                raise ValueError("the task out buffer must be returned")
            if args.task=='permute_copy' and value.untyped_storage().data_ptr()==call.bound['input'].untyped_storage().data_ptr():
                raise ValueError("permute_copy requires a fresh output allocation")
            state['output']=value
        def outputs():
            from torch.utils._pytree import tree_flatten
            import torch
            tensors=tuple(x for x in tree_flatten((call.args,call.kwargs))[0] if isinstance(x,torch.Tensor))
            return state['output'],tensors
        return PreparedCall(run,outputs,reset=lambda:restore((call.args,call.kwargs),initial),device=args.device)
    def source(case,call,env):return prepare(reference,call)
    def candidate(case,call,env):
        metadata={'candidate':str(args.program.resolve()),'suite':str(args.suite.resolve()),
                  'reference':str(args.reference.resolve())}
        if args.language=='intent':
            intent_environment(env)
            import torch
            from .corpus.agent_tritonbench.program import ProgramContext,load_program
            context=ProgramContext(args.compiler,args.program.parent,language='intent',target=args.target,
                                   device=torch.device(args.device).index or 0)
            module=load_program(args.program,language='intent')
            function=module.build(context)
            metadata['artifacts']=[str(artifact.cache_directory) for artifact in context.generated.values()]
        else:
            from .corpus.agent_tritonbench.policy import validate_program
            validate_program(args.program,language='triton')
            function=load(str(args.program)+':build')(SimpleNamespace(language='triton',target=args.target,device=args.device))
        prepared=prepare(function,call)
        prepared.metadata.update(metadata)
        return prepared
    tolerance=tasks.tolerance(task,suite)
    def check(actual,expected):
        from .numeric import compare_outputs
        compare_outputs(actual,expected,tolerance)
    name=args.language+(':'+args.target if args.language=='intent' else '')
    work=Workload('agent/'+args.task,[Case('profile-'+str(task['input_index']))],inputs,
                  {name:Implementation(name,candidate),'reference':Implementation('reference',source)},
                  oracle=Implementation('oracle',source),check=check,family='agent/'+row.get('family','tritonbench'))
    rows=benchmark(work,environment=env,measurement=Measurement(args.timing,args.samples,args.warmup),
                   baseline='reference',seed=suite['input_seed'])
    print(table(rows));directory=save(rows,args.output,command=getattr(args,'command_line',None))
    print(directory.resolve())
    return int(any(r.status not in {'passed','unchecked'} for r in rows))


def generate(args):
    tasks,suite,task,row=task_data(args)
    material=tasks.invocation(args.reference,row,task,suite,device='cpu')
    reference=tasks.reference(args.reference,row)
    row.update(task)
    row.update(reference_signature=row['entry']+str(inspect.signature(reference)),invocation=material.metadata(),
               return_contract=tasks.return_contract(material.call(reference)),
               timing=task.get('timing',suite['timing']))
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Agent generation requires a new experiment output directory")
    if args.codex:
        intent_environment(Environment(intent_root=args.intent_root))
        from .corpus.agent_tritonbench.generation import generate_trial
        if not args.state_root or not args.triton_ref or not args.intent_root:
            raise ValueError('the isolated Codex preset requires --state-root, --triton-ref and --intent-root')
        (args.state_root/'candidates').mkdir(parents=True,exist_ok=True)
        (args.state_root/'codex').mkdir(exist_ok=True)
        config=SimpleNamespace(state_root=args.state_root,project=args.intent_root,triton_ref=args.triton_ref,
            reference=args.reference,suite=suite,stop=threading.Event(),codex=args.codex,output=args.output)
        result=generate_trial(config,row,args.language)
    else:
        # An explicit external command is a trusted caller-owned generator, not a sandbox preset.
        task_dir=args.output/'task';task_dir.mkdir()
        prompt=tasks.description(args.reference,row)+'\n\nPublic invocation contract:\n'+json.dumps({
            'signature':row['reference_signature'],'inputs':row['invocation'],'outputs':row['return_contract'],
            'language':args.language,'tolerance':suite['tolerances'][task['tolerance']]},indent=2)
        (task_dir/'TASK.md').write_text(prompt)
        command=json.loads(args.command)
        if not isinstance(command,list) or not all(isinstance(x,str) for x in command):
            raise ValueError('--command must be a JSON array of argv strings')
        command=[x.replace('{task_dir}',str(task_dir.resolve())).replace('{output}',str((task_dir/'candidate.py').resolve())) for x in command]
        with (args.output/'generation.log').open('w') as log:
            proc=subprocess.run(command,cwd=task_dir,stdout=log,stderr=subprocess.STDOUT,timeout=suite['agent_seconds'])
        result={'task':args.task,'language':args.language,'status':'submitted' if proc.returncode==0 and (task_dir/'candidate.py').is_file() else 'generation_failed',
                'program':str(task_dir/'candidate.py'),'returncode':proc.returncode,'policy':'external command; one invocation; no evaluation feedback'}
    (args.output/'generation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return int(result['status']!='submitted')


def main(argv):
    parser=argparse.ArgumentParser(prog='intent-bench agent')
    sub=parser.add_subparsers(dest='action',required=True)
    default_suite=Path(__file__).parent/'corpus/agent_tritonbench/suite-100.json'
    for action in ('generate','evaluate'):
        p=sub.add_parser(action)
        p.add_argument('--reference',type=Path,required=True)
        p.add_argument('--suite',type=Path,default=default_suite)
        p.add_argument('--task',required=True)
        p.add_argument('--language',choices=('intent','triton'),required=True)
        p.add_argument('--intent-root',type=Path)
        p.add_argument('--output',type=Path,required=True)
        if action=='evaluate':
            p.add_argument('--program',type=Path,required=True)
            p.add_argument('--compiler')
            p.add_argument('--target',choices=('triton','cutile'),default='triton')
            p.add_argument('--device',default='cuda:0')
            p.add_argument('--timing',choices=('cuda_event','cuda_graph','host'),default='cuda_event')
            p.add_argument('--samples',type=int)
            p.add_argument('--warmup',type=int)
        else:
            choice=p.add_mutually_exclusive_group(required=True)
            choice.add_argument('--command',help='JSON argv, with {task_dir} and {output} placeholders')
            choice.add_argument('--codex',type=Path)
            p.add_argument('--state-root',type=Path)
            p.add_argument('--triton-ref',type=Path)
    args=parser.parse_args(argv)
    args.command_line=['intent-bench','agent',*argv]
    from .output import output_root, run_output
    root = output_root()
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(root/"experiments"):
        parser.error("Agent output must be under the configured output_root/experiments")
    with run_output(Environment(name="agent", output_root=root), "agent/"+args.task, args.output):
        return generate(args) if args.action=='generate' else evaluate(args)
