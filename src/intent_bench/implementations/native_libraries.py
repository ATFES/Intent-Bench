"""GGML and CNNL bindings; execution transport is owned by WorkerSession."""
from pathlib import Path

from ..execution import WorkerSession
from ..model import EnvironmentUnavailable, PreparedCall


def pack_q4(inputs,env):
    required=('ggml_source','ggml_libraries')
    if any(k not in env.options for k in required):raise EnvironmentUnavailable('Q4_K workload input encoding requires ggml_source and ggml_libraries options')
    root=Path(__file__).parents[1]/'native'
    request={'operation':'q4_projection','source':env.options['ggml_source'],'libraries':env.options['ggml_libraries'],
             'threads':env.threads,'cxx':env.options.get('cxx','c++')}
    session=WorkerSession(env,'ggml_worker.py',[root/'cnnl_worker.py',root/'ggml_worker.py',root/'ggml_ops.cpp'],request,inputs)
    try:return session.result('packed')
    finally:session.close()


def cnnl(name,case,inputs,env):
    root=Path(__file__).parents[1]/"native"
    request={"operation":name,"arity":len(inputs),"device":int(env.device.split(":")[1]),"neuware":env.options.get("neuware","/usr/local/neuware")}
    session=WorkerSession(env,"cnnl_worker.py",[root/"cnnl_worker.py",root/"cnnl_ops.cpp"],request,inputs)
    return PreparedCall(lambda:session.call("run"),session.result,close=session.close,
        native_measure=lambda settings:session.call("measure",samples=settings.samples,warmup=settings.warmup),
        device=session.device(f"mlu:{request['device']}"),timing="cnrt_notifier",
        metadata={**session.metadata(),
                  "scope_note":"Complete CNNL call; descriptors, workspace allocation and transport excluded"})


def ggml(name,case,inputs,env,*,reference=False):
    required=("ggml_source","ggml_libraries")
    missing=[key for key in required if key not in env.options]
    if missing:raise EnvironmentUnavailable("GGML requires environment options: "+", ".join(missing))
    root=Path(__file__).parents[1]/"native"
    request={"operation":name,"source":env.options["ggml_source"],"libraries":env.options["ggml_libraries"],
             "threads":env.threads,"reference":reference,"cxx":env.options.get("cxx","c++")}
    session=WorkerSession(env,"ggml_worker.py",[root/"cnnl_worker.py",root/"ggml_worker.py",root/"ggml_ops.cpp"],request,inputs)
    metadata={**session.metadata(),
              "execution":"prepared native graph plan and persistent threadpool" if name not in {"q4_dot","q8_quantize"}
                          else "native quantization/dot primitive",
              "weights":"preencoded Q4_K input bytes" if name=="q4_projection_packed" else
                        "Q4_K weights encoded once outside timing" if name.startswith("q4") else "f32",
              "activation_quantization":"inside graph call" if name in {"q4_projection","q4_projection_packed"}
                                        else "preencoded" if name=="q4_dot" else "not applicable"}
    if name=="gated_delta":
        metadata.update(library="GGML CPU ggml_gated_delta_net", weights="not applicable",
            scope_note="BF16 inputs; FP32 conversion, grouped Q/K expansion, native recurrence, "
                       "BF16 output conversion and final-state transpose all inside the graph call.")
    def measure(settings):
        metadata['cache_policy']='flush_cpu_64m' if settings.cache=='flush' else 'warm'
        return session.call("measure",samples=settings.samples,warmup=settings.warmup,cache=settings.cache)
    return PreparedCall(lambda:session.call("run"),session.result,close=session.close,
        native_measure=measure,
        device=session.device("cpu"),timing="cpu_native_monotonic",
        metadata=metadata)
