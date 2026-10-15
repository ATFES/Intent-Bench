"""Persistent native workers: transport and compilation happen outside measurement."""
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

from ..model import EnvironmentUnavailable, PreparedCall


class NativeSession:
    def __init__(self,host,worker,files,request,inputs,python="python3"):
        self.host=host;self.process=None;self.remote=None
        root=Path(os.environ.get("XDG_CACHE_HOME",Path.home()/".cache"))/"intent-bench/remote"
        root.mkdir(parents=True,exist_ok=True)
        self.local=Path(tempfile.mkdtemp(prefix="run-",dir=root))
        try:
            import shutil
            for file in files:
                file=Path(file)
                if file.is_dir():shutil.copytree(file,self.local/file.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                else:shutil.copy2(file,self.local/file.name)
            for index,value in enumerate(inputs):np.save(self.local/f"input{index}.npy",value,allow_pickle=False)
            (self.local/"request.json").write_text(json.dumps(request))
            self.remote=subprocess.check_output(["ssh","-o","BatchMode=yes",host,"mktemp -d /tmp/intent-bench.XXXXXX"],text=True,timeout=30).strip()
            subprocess.run(["scp","-qr",*[str(p) for p in self.local.iterdir()],host+":"+self.remote+"/"],check=True,timeout=request.get('transfer_timeout',600))
            command=shlex.join([python,"-u",self.remote+"/"+worker,self.remote+"/request.json"])
            self.log=(self.local/"worker.log").open("w")
            self.process=subprocess.Popen(["ssh","-o","BatchMode=yes",host,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
            self.info=self.receive()
            if not self.info.get("ready"):raise RuntimeError("worker did not become ready")
        except Exception:
            self.close();raise

    def receive(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(f"native worker exited; log: {self.local/'worker.log'}")
        result=json.loads(line)
        if "error" in result:raise RuntimeError(result["error"])
        return result

    def call(self,action,**options):
        self.process.stdin.write(json.dumps({"action":action,**options})+"\n");self.process.stdin.flush()
        return self.receive().get("value")

    def metadata(self):
        return {
            **self.info.get("environment", {}),
            "native_compile_seconds": self.info["compile_seconds"],
            "artifacts": str(self.local),
        }

    def result(self,action='result'):
        def decode(data):
            if 'items' in data:return tuple(decode(item) for item in data['items'])
            raw=base64.b64decode(data['data'])
            if data.get('torch_dtype'):
                import torch
                return torch.frombuffer(bytearray(raw),dtype=getattr(torch,data['torch_dtype'])).reshape(data['shape']).clone()
            return np.frombuffer(raw,dtype=data['dtype']).reshape(data['shape']).copy()
        return decode(self.call(action))

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"action":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=15)
                except (BrokenPipeError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
            self.process.stdin.close();self.process.stdout.close();self.log.close()
        if self.remote:
            # Only the directory returned by our own mktemp is removed.
            subprocess.run(["ssh","-o","BatchMode=yes",self.host,shlex.join(["rm","-rf","--",self.remote])],timeout=30,check=True)
            self.remote=None


def pack_q4(inputs,env):
    required=('host','ggml_source','ggml_libraries')
    if any(k not in env.options for k in required):raise EnvironmentUnavailable('Q4_K workload input encoding requires host, ggml_source and ggml_libraries options')
    root=Path(__file__).parents[1]/'native'
    request={'operation':'q4_projection','source':env.options['ggml_source'],'libraries':env.options['ggml_libraries'],
             'threads':env.threads,'cxx':env.options.get('cxx','c++')}
    session=NativeSession(env.options['host'],'ggml_worker.py',[root/'cnnl_worker.py',root/'ggml_worker.py',root/'ggml_ops.cpp'],request,inputs,env.options.get('python','python3'))
    try:return session.result('packed')
    finally:session.close()


def cnnl(name,case,inputs,env):
    host=env.options.get("host")
    if not host:raise EnvironmentUnavailable("CNNL requires --option host=\"SSH_ALIAS\"")
    root=Path(__file__).parents[1]/"native"
    request={"operation":name,"arity":len(inputs),"device":env.options.get("device",0),"neuware":env.options.get("neuware","/usr/local/neuware")}
    session=NativeSession(host,"cnnl_worker.py",[root/"cnnl_worker.py",root/"cnnl_ops.cpp"],request,inputs,env.options.get("python","python3"))
    return PreparedCall(lambda:session.call("run"),session.result,close=session.close,
        native_measure=lambda settings:session.call("measure",samples=settings.samples,warmup=settings.warmup),
        device=f"{host}/mlu:{request['device']}",timing="cnrt_notifier",
        metadata={**session.metadata(),
                  "scope_note":"Complete CNNL call; descriptors, workspace allocation and transport excluded"})


def ggml(name,case,inputs,env,*,reference=False):
    required=("host","ggml_source","ggml_libraries")
    missing=[key for key in required if key not in env.options]
    if missing:raise EnvironmentUnavailable("GGML requires environment options: "+", ".join(missing))
    host=env.options["host"];root=Path(__file__).parents[1]/"native"
    request={"operation":name,"source":env.options["ggml_source"],"libraries":env.options["ggml_libraries"],
             "threads":env.threads,"reference":reference,"cxx":env.options.get("cxx","c++")}
    session=NativeSession(host,"ggml_worker.py",[root/"cnnl_worker.py",root/"ggml_worker.py",root/"ggml_ops.cpp"],request,inputs,env.options.get("python","python3"))
    return PreparedCall(lambda:session.call("run"),session.result,close=session.close,
        native_measure=lambda settings:session.call("measure",samples=settings.samples,warmup=settings.warmup),
        device=host+"/cpu",timing="cpu_native_monotonic",
        metadata={**session.metadata(),
                  "weights":"Q4_K weights encoded once outside timing" if name.startswith("q4") else "f32",
                  "activation_quantization":"inside graph call" if name=="q4_projection" else "preencoded" if name=="q4_dot" else "not applicable"})
