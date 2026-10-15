from intent_bench.corpus.gpu.registry import BY_PROVIDER as GPU
from intent_bench.corpus.cpu.registry import BY_PROVIDER as CPU
from intent_bench.corpus.mlu.registry import BY_PROVIDER as MLU


BY_PROVIDER = {**GPU, **CPU, **MLU}
PROVIDER_GROUPS = {
    **dict.fromkeys(GPU, "gpu"),
    **dict.fromkeys(CPU, "cpu"),
    **dict.fromkeys(MLU, "mlu"),
}
