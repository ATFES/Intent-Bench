"""Local and SSH process transport. No operator or compiler backend logic."""
import base64
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace

import numpy as np


class Workspace:
    def __init__(self, environment):
        self.host = environment.host or environment.options.get("host")
        self.transport = environment.transport or ("ssh" if self.host else "local")
        self.python = environment.python or environment.options.get("python") or ("python3" if self.host else sys.executable)
        if self.transport == "ssh" and not self.host:
            raise ValueError("SSH execution requires a host")
        if self.transport == "local" and self.host:
            raise ValueError("local execution must not specify a host")
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"intent-bench/workers"
        root.mkdir(parents=True, exist_ok=True)
        self.local = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        self.remote = None

    def copy(self, files):
        for item in files:
            item = Path(item)
            if item.is_dir():
                shutil.copytree(item, self.local/item.name, ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            else:
                shutil.copy2(item, self.local/item.name)

    def stage(self, timeout=600):
        if self.transport == "local":
            return str(self.local)
        self.remote = subprocess.check_output(["ssh","-o","BatchMode=yes",self.host,
            "mktemp -d /tmp/intent-bench.XXXXXX"], text=True, timeout=30).strip()
        subprocess.run(["scp","-qr",*[str(p) for p in self.local.iterdir()],self.host+":"+self.remote+"/"],
                       check=True, timeout=timeout)
        return self.remote

    def start(self, arguments, **kwargs):
        if self.transport == "local":
            return subprocess.Popen(arguments, cwd=self.local, **kwargs)
        command = "cd " + shlex.quote(self.remote) + " && " + shlex.join(arguments)
        return subprocess.Popen(["ssh","-o","BatchMode=yes",self.host,command], **kwargs)

    def fetch(self, name, destination):
        if self.transport == "local":
            shutil.copytree(self.local/name, destination, dirs_exist_ok=True)
        else:
            received=self.local/'received'
            received.mkdir(exist_ok=True)
            subprocess.run(["scp","-qr",self.host+":"+self.remote+"/"+name,str(received)],check=True,timeout=600)
            shutil.copytree(received/name,destination,dirs_exist_ok=True)

    def device(self, name):
        return self.host+"/"+name if self.transport == "ssh" else name

    def close(self):
        if self.remote:
            subprocess.run(["ssh","-o","BatchMode=yes",self.host,
                shlex.join(["rm","-rf","--",self.remote])], timeout=30, check=True)
            self.remote = None


class WorkerSession:
    def __init__(self, environment, worker, files, request, inputs):
        self.workspace = Workspace(environment)
        self.local = self.workspace.local
        self.process = None
        self.log = None
        try:
            self.workspace.copy(files)
            for index,value in enumerate(inputs):
                np.save(self.local/f"input{index}.npy", value, allow_pickle=False)
            (self.local/"request.json").write_text(json.dumps(request))
            location = self.workspace.stage(request.get("transfer_timeout",600))
            self.log = (self.local/"worker.log").open("w")
            self.process = self.workspace.start([self.workspace.python,"-u",location+"/"+worker,location+"/request.json"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
            self.info = self.receive()
            if not self.info.get("ready"):
                raise RuntimeError("worker did not become ready")
        except Exception:
            self.close()
            raise

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited; log: {self.local/'worker.log'}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def call(self, action, **options):
        self.process.stdin.write(json.dumps({"action":action,**options})+"\n")
        self.process.stdin.flush()
        return self.receive().get("value")

    def metadata(self):
        return {**self.info.get("environment",{}), "transport":self.workspace.transport,
                "native_compile_seconds":self.info["compile_seconds"], "artifacts":str(self.local)}

    def device(self, name):
        return self.workspace.device(name)

    def result(self, action="result"):
        def decode(data):
            if "items" in data:
                return tuple(decode(item) for item in data["items"])
            raw = base64.b64decode(data["data"])
            if data.get("torch_dtype"):
                import torch
                return torch.frombuffer(bytearray(raw),dtype=getattr(torch,data["torch_dtype"])).reshape(data["shape"]).clone()
            return np.frombuffer(raw,dtype=data["dtype"]).reshape(data["shape"]).copy()
        return decode(self.call(action))

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"action":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=15)
                except (BrokenPipeError,subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        if self.log is not None:
            self.log.close()
        self.workspace.close()


def dispatch_cli(arguments, args, environment):
    """Run Python operators in the chosen host/interpreter; native adapters use workers."""
    native = args.command == 'run' and (
        getattr(args,'suite',None) in ('mlu','rvv') or
        (getattr(args,'workload',None) or '').startswith(('mlu/','rvv/','corpus/bangc/','corpus/weft/')))
    remote = environment.transport == 'ssh' and not native
    interpreter = environment.controller_python or (environment.python if environment.transport != 'ssh' else None)
    local_python = (not remote and interpreter and
                    str(Path(shutil.which(interpreter) or interpreter).absolute()) != str(Path(sys.executable).absolute()))
    if not remote and not local_python:
        return None
    workspace = Workspace(environment)
    try:
        workspace.copy([Path(__file__).parent])
        if local_python:
            process_env = dict(os.environ)
            process_env['PYTHONPATH'] = str(workspace.local)+os.pathsep+process_env.get('PYTHONPATH','')
            return subprocess.call([interpreter,'-m','intent_bench.cli',*arguments],env=process_env)
        from .environments import describe_environment
        from .results import read,save,table
        forwarded=[]
        skip=False
        for item in arguments:
            if skip:skip=False;continue
            if item in ('--env','--output'):skip=True;continue
            if item.startswith(('--env=','--output=')):continue
            forwarded.append(item)
        workload = getattr(args,'workload',None)
        if workload:
            filename,separator,symbol=workload.partition(':')
            if Path(filename).is_file():
                workspace.copy([Path(filename).resolve()])
                forwarded[forwarded.index(workload)]='./'+Path(filename).name+(separator+symbol if separator else '')
        local_options={key:value for key,value in environment.options.items() if key not in ('host','python')}
        local_env=replace(environment,transport='local',host=None,python=None,controller_python=None,options=local_options)
        (workspace.local/'environment.json').write_text(json.dumps(describe_environment(local_env)))
        workspace.stage()
        output=Path(args.output or Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'intent-bench/runs'/workspace.local.name)
        output.mkdir(parents=True,exist_ok=True)
        with (output/'worker.log').open('w') as log:
            process=workspace.start([workspace.python,'-m','intent_bench.cli',*forwarded,
                                     '--env','environment.json','--output','output'],stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        workspace.fetch('output',output)
        rows=read(output)
        for row in rows:
            row['device']=workspace.device(row['device'])
            row.setdefault('metadata',{}).update(transport='ssh',host=workspace.host)
        save(rows,output,command=['intent-bench',*arguments])
        print(table(rows));print(f'Results: {output.resolve()}')
        return code
    finally:
        workspace.close()
