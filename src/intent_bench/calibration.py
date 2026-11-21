"""Measure a device profile on its execution host. Data stays in the requested output."""
import argparse
from pathlib import Path
import json
import os
import platform
import statistics
import subprocess
from collections import defaultdict

def summary(samples):
    measured=samples[1:]
    return statistics.median(measured),f"range [{min(measured):.6g}, {max(measured):.6g}], n={len(measured)}"

def cpu(root, args):
    src=Path(__file__).with_name("native")/"calibrate_cpu.c"
    binary=root/"calibrate_cpu"
    cmd=[args.cc,"-O3","-fopenmp",*args.cflag,str(src),"-o",str(binary)]
    subprocess.run(cmd,check=True)
    raw=subprocess.check_output([binary,str(args.threads)],text=True)
    by_name=defaultdict(list)
    for line in raw.splitlines():
        name,value=line.split();by_name[name].append(float(value))
    return by_name,subprocess.check_output([args.cc,"--version"],text=True).splitlines()[0], " ".join(v for v in args.cflag if v.startswith(("-march=", "-mabi=")))

def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend",choices=["cpu","cuda","mlu"]);p.add_argument("output",type=Path)
    p.add_argument("--cc",default="cc");p.add_argument("--cflag",action="append",default=[])
    p.add_argument("--threads",type=int,default=1);p.add_argument("--cpus");p.add_argument("--profile",required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.cpus:os.sched_setaffinity(0,{int(i) for i in a.cpus.split(",")})
    if a.backend=="cpu":
        data,tool,flags=cpu(root,a)
        info = dict(line.split(":", 1) for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines() if ":" in line)
        info = {key.strip(): value.strip() for key,value in info.items()}
        device=info.get("model name") or (info.get("mvendorid","")+"/"+info.get("marchid",""))
        isa=platform.machine()+" "+flags
        dram_mib=1024 if platform.machine()=="x86_64" else 256
        conditions=(f"affinity={sorted(os.sched_getaffinity(0))}, threads={a.threads}, FP32; "
                    f"private {16*a.threads} KiB, mid {512*a.threads} KiB, shared 16 MiB, DRAM {dram_mib} MiB; "
                    "paired copy uses binary16 32 MiB sparse footprint; unmanaged clocks")
    else:
        from . import calibration_device
        data,tool,device,isa,conditions=calibration_device.probe(a.backend,root)
    for op in ("read","write"):
        for step in (1,16):
            key=f"{op}_{step}"
            data[key]=data[key+"_dram_bw"]
    facts={}
    for k,samples in data.items():
        val,error=summary(samples)
        unit="B/ns" if k.startswith(("read_","write_")) or k.endswith("_bw") else "lane/ns" if k=="compute" else "FLOP/ns" if k=="vector" else "ns"
        facts[k]=dict(value=val,unit=unit,source=dict(device=device,isa=isa,toolchain=tool,conditions=conditions,benchmark="intent_bench.calibration/"+k,error=error))
    (root/"samples.json").write_text(json.dumps(data,indent=2))
    (root/(a.profile+".calibration.json")).write_text(json.dumps(dict(version="3",facts=facts),indent=2)+"\n")
if __name__=="__main__":run()
