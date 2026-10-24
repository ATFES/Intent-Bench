"""Native AOTInductor comparisons with the workload's exact arithmetic."""
from pathlib import Path
import tempfile
import time

from ..output import output_root


def prepare(name, arguments, environment):
    import torch
    from . import aten_cpu

    if name != "dropout":
        raise NotImplementedError(("AOTInductor CPU", name))
    x, seed, probability, inverse_keep = arguments
    if environment.device != "cpu" or x.dtype != torch.float16 or x.ndim != 2:
        raise ValueError("xorshift dropout comparison requires a CPU FP16 matrix")
    torch.set_num_threads(environment.threads)

    class Dropout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(inverse_keep, dtype=x.dtype))

        def forward(self, value, output):
            offsets = torch.arange(value.numel(), dtype=torch.int32).reshape(value.shape)
            hashed = offsets * 1103515245 + seed
            hashed = hashed ^ (hashed >> 16)
            hashed = hashed ^ (hashed << 8)
            hashed = hashed ^ (hashed >> 4)
            random = (hashed & 0x7fffffff).float() / 2147483647.0
            output.copy_(torch.where(random > probability, value * self.scale, 0.0))
            return output

    parent = output_root(environment.output_root)/"build/native"
    parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="inductor-cpu-", dir=parent))
    output = torch.empty_like(x)
    started = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(Dropout(), (x, output))
        package = torch._inductor.aoti_compile_and_package(
            exported, package_path=str(build/"model.pt2"),
            inductor_configs={"cpp.threads": environment.threads})
    seconds = time.perf_counter() - started
    call = aten_cpu.prepare("aoti", (x, output), environment, package=package)
    call.metadata.update(
        library="PyTorch AOTInductor CPU fused xorshift dropout",
        package=str(package), preparation_stages_seconds={"native_compile": seconds},
        scope_note="Exact deterministic xorshift and FP16 scale rounding; prebound output. "
                   "Native C++ package execution is timed; export, compilation, loading and Python are outside.")
    return call
