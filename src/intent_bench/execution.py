"""Local and SSH process transport. No operator or compiler backend logic."""
import base64
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from dataclasses import replace

import numpy as np


class Workspace:
    def __init__(self, environment, *, directory=None):
        self.host = environment.host or environment.options.get("host")
        self.transport = environment.transport or ("ssh" if self.host else "local")
        self.python = environment.python or environment.options.get("python") or ("python3" if self.host else sys.executable)
        self.launcher = tuple(environment.launcher)
        if self.transport == "ssh" and not self.host:
            raise ValueError("SSH execution requires a host")
        if self.transport == "local" and self.host:
            raise ValueError("local execution must not specify a host")
        from .output import worker_root
        root = Path(directory) if directory is not None else worker_root()
        root.mkdir(parents=True, exist_ok=True)
        self.local = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        self.remote = None
        self.remote_root = environment.remote_output_root

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
        if not self.remote_root:
            raise ValueError("SSH environments require remote_output_root")
        work = self.remote_root.rstrip('/')+'/work'
        command = 'mkdir -p '+shlex.quote(work+'/tmp')+' && mktemp -d '+shlex.quote(work+'/worker-XXXXXX')
        self.remote = subprocess.check_output(["ssh","-o","BatchMode=yes",self.host, command],
                                             text=True, timeout=30).strip()
        files = [str(p) for p in self.local.iterdir()]
        if files:
            subprocess.run(["scp","-qr",*files,self.host+":"+self.remote+"/"],
                           check=True, timeout=timeout)
        return self.remote

    def start(self, arguments, **kwargs):
        arguments = [*self.launcher, *arguments]
        if self.transport == "local":
            return subprocess.Popen(arguments, cwd=self.local, **kwargs)
        command = "cd " + shlex.quote(self.remote) + " && " + shlex.join([
            "env", "PYTHONDONTWRITEBYTECODE=1", "TMPDIR="+self.remote_root.rstrip('/')+"/work/tmp", *arguments])
        return subprocess.Popen(["ssh","-o","BatchMode=yes",self.host,command], **kwargs)

    def fetch(self, name, destination):
        if self.transport == "local":
            shutil.copytree(self.local/name, destination, dirs_exist_ok=True)
        else:
            received=self.local/'received'
            received.mkdir(exist_ok=True)
            subprocess.run(["scp","-qr",self.host+":"+self.remote+"/"+name,str(received)],check=True,timeout=600)
            shutil.copytree(received/name,destination,dirs_exist_ok=True)

    def run(self, arguments, *, timeout, log, environment=None):
        """Bound the selected worker and its descendants on either transport."""
        if self.transport == "ssh":
            arguments = ["timeout", "--signal=TERM", "--kill-after=5", str(timeout), *arguments]
        with Path(log).open("w") as stream:
            process = self.start(arguments, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=environment)
            try:
                return process.wait(timeout=timeout + (10 if self.transport == "ssh" else 0))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124

    def device(self, name):
        return self.host+"/"+name if self.transport == "ssh" else name

    def close(self, *, keep_local=False):
        if self.remote:
            subprocess.run(["ssh","-o","BatchMode=yes",self.host,
                shlex.join(["rm","-rf","--",self.remote])], timeout=30, check=True)
            self.remote = None
        if not keep_local:
            shutil.rmtree(self.local)


class WorkerSession:
    def __init__(self, environment, worker, files, request, inputs):
        self.workspace = Workspace(environment, directory=Path(os.environ["INTENT_BENCH_RUN_DIR"])/"artifacts")
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
        self.workspace.close(keep_local=True)


def dispatch_request(request):
    """Transport the same request; provider and workload names never select transport."""
    from .model import Result
    from .output import output_root, run_output
    from .results import read, save
    from .service import RunOutcome
    environment = request.environment
    remote = environment.transport == "ssh"
    root = output_root(environment.output_root)
    temporary = root/"work"/"tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    with run_output(environment, request.workload, request.output, execution=False) as output:
        workspace = Workspace(environment, directory=root/"work")
        rows = []
        try:
            # The selected interpreter must have Bench and Intent installed.
            # Only explicitly supplied user workloads are transferred.
            workload = request.workload
            filename, separator, symbol = workload.partition(":")
            if Path(filename).is_file():
                workspace.copy([Path(filename).resolve()])
                workload = "./"+Path(filename).name+(separator+symbol if separator else "")
            local_env = replace(environment, transport="local", host=None, python=None,
                                output_root=Path(environment.remote_output_root) if remote else output_root(environment.output_root))
            location = workspace.stage()
            forwarded = replace(request, workload=workload, environment=local_env,
                                output=Path(location)/"output")
            payload = workspace.local/"request.json"
            payload.write_text(json.dumps(forwarded.record()))
            if remote:
                subprocess.run(["scp","-q",str(payload),workspace.host+":"+location+"/request.json"],
                               check=True,timeout=30)
            code = workspace.run([workspace.python,"-m","intent_bench.worker",location+"/request.json"],
                                 timeout=request.timeout, log=output/"worker.log",
                                 environment={**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "TMPDIR":str(temporary)})
            try:
                workspace.fetch("output", output)
                rows = read(output)
                def relocate(value):
                    if isinstance(value, str) and value.startswith(location+"/output/"):
                        return str(output)+value[len(location+"/output"):]
                    if isinstance(value, list):
                        return [relocate(item) for item in value]
                    if isinstance(value, dict):
                        return {key:relocate(item) for key,item in value.items()}
                    return value
                rows = relocate(rows)
            except (FileNotFoundError, subprocess.CalledProcessError):
                rows = [Result(request.workload,"requested",case,{},name,status="worker_failed",
                               stage="worker",device=environment.device,
                               error=f"worker exit {code}; log: {output/'worker.log'}")
                        for case in request.cases or ("load",) for name in request.implementations]
            for row in rows:
                if isinstance(row,dict):
                    if code and row.get("status") == "pending":
                        row["status"] = "worker_failed"
                        row["error"] = f"worker exit {code} during {row['stage']}; log: {output/'worker.log'}"
                    row["device"] = workspace.device(row["device"])
                    row.setdefault("metadata",{}).update(transport=workspace.transport, worker_log=str(output/"worker.log"))
            if code and not any((r["status"] if isinstance(r,dict) else r.status) not in
                                {"passed","unchecked","generated","compiled","executed"} for r in rows):
                rows.append(Result(request.workload,"requested","worker",{},"worker",
                                   status="worker_failed",stage="worker",device=environment.device,
                                   error=f"worker exit {code}; log: {output/'worker.log'}"))
            if request.oracle_environment is not None and request.kind == "workload":
                from .oracle import validate_request
                rows = validate_request(request, output, rows)
            save(rows, output, command=request.record())
            return RunOutcome(rows, output)
        except Exception as error:
            rows = [Result(request.workload,"requested",case,{},name,status="worker_failed",
                           stage="transport",device=environment.device,error=str(error),
                           traceback=traceback.format_exc())
                    for case in request.cases or ("load",) for name in request.implementations]
            save(rows, output, command=request.record())
            return RunOutcome(rows, output)
        finally:
            try:
                workspace.close()
            except Exception as error:
                rows.append(Result(request.workload,"requested","cleanup",{},"worker",status="cleanup_failed",
                                   stage="cleanup",device=environment.device,error=str(error)))
                save(rows, output, command=request.record())
