# Source attribution

IntentDSL Bench's framework and authored adapters are Apache-2.0 licensed, as in
the originating IntentDSL project. Vendored implementations retain their own
headers and licenses; the framework license does not replace those terms.

The implementation directories below are under `src/intent_bench/corpus/gpu/baselines`.
The preserved license texts are included in `src/intent_bench/licenses` and in the wheel.

| Directory | Upstream | License |
|---|---|---|
| `triton/triton` | [Triton](https://github.com/triton-lang/triton) | MIT |
| `triton/flash-attention` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause |
| `triton/liger-kernel` | [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | BSD-2-Clause |
| `triton/xformers` | [xFormers](https://github.com/facebookresearch/xformers) | BSD-3-Clause |
| `triton/tritonbench` | [Tritonbench](https://github.com/meta-pytorch/tritonbench) | BSD-3-Clause |
| `triton/meta-applied-ai` | [Applied AI](https://github.com/meta-pytorch/applied-ai) | BSD-3-Clause |
| `triton/state-spaces-mamba` | [Mamba](https://github.com/state-spaces/mamba) | Apache-2.0 |
| `triton/fla` | [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | MIT |
| `triton/vllm` | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 |
| `triton/flag-gems` | [FlagGems](https://github.com/FlagOpen/FlagGems) | Apache-2.0 |
| `cutile/cutile-python` | [cuTile Python](https://github.com/NVIDIA/cutile-python) | Apache-2.0 |
| `cutile/tilegym` | [TileGym](https://github.com/NVIDIA/TileGym) | MIT (operator sources) |

Baseline runtime wrappers, namespace loaders, workload adapters, and independent
Mojo references were adapted from IntentDSL's evaluation code. Their imports,
input ownership and runtime bindings were changed during extraction. Upstream
kernel headers are preserved. Mojo implementations using MAX call an external
installed SDK; GGML and CNNL are likewise external libraries, not bundled binaries.

The `archive` directory contains historical material, including older CUDA
sources and Agent candidates. It is not installed as part of the tool. Its files
retain their original notices and recorded provenance. The migration index maps
original paths to their retained destinations without hashes or checksums.
