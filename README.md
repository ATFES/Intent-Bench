# IntentDSL Bench

独立的算子正确性检查和性能比较工具。默认看延迟，需要时加一个吞吐指标。输入、实现、数值参考、执行设备和测量方式分别选择；IntentDSL 是可以参与比较的实现之一。

[English](README.en.md) · [第三方来源与许可证](THIRD_PARTY.md)

工具有三个等价入口：终端 `intent-bench`、Python `benchmark(...)`、Agent 的 `intent-bench-mcp`。插件把 bench 与主线已有的手册／编译器 MCP 放在一起；算子执行复用同一个引擎。

## 安装与运行

```sh
python -m pip install -e .
intent-bench list
intent-bench describe softmax
intent-bench run softmax --case 64x1024-float16 --impl torch,triton,cutile --baseline triton --device cuda:0
intent-bench run examples/custom.py:bench
intent-bench run cpu/adamw --impl mojo --option 'mojo="/path/to/mojo"'
intent-bench run advanced/scaled_fp8_gemm --impl mojo,intent:mojo --threads 8 --intent-root /path/to/intentdsl
```

核心仅依赖 NumPy。Torch、Triton、cuTile、Mojo、Intent、GGML、CNNL 随所选实现安装；`list` 不加载这些 SDK。使用当前 Python 环境运行 Python 实现。原生 Mojo 可以通过 `INTENT_BENCH_MOJO` 选择编译器。

```sh
intent-bench run softmax --case 64x1024-float16 \
  --impl intent:triton,intent:cutile,triton,cutile --baseline triton \
  --intent-root /path/to/intentdsl --compiler /path/to/intent-compile --device cuda:0
intent-bench run softmax --case 64x1024-float16 --impl torch \
  --param 'rows=[1,64,512]' --param 'columns=[128,1024,4096]'
intent-bench check --intent-root /path/to/intentdsl --suite mainline --target triton --stage run
intent-bench compare runs/before runs/after
intent-bench report runs/latest
```

`check` 调用主线原有作者程序，分别报告源码生成、native 编译或执行；`executed` 不表示已做数值对照。正式数值与性能比较使用 `run` 的 workload。

## 选择实现和机器

`--impl` 选择参与比较的实现，`--env` 选择运行环境。二者独立：

| 选择 | 含义 |
|---|---|
| `triton` / `cutile` | 手写的原生 GPU 实现 |
| `intent:triton` / `intent:cutile` | 将 Intent 编译到指定后端 |
| `mojo` / `intent:mojo` | 原生 Mojo / Intent→Mojo |
| `ggml` / `weft` / `intent:weft` | 独立 GGML / 原生 Weft / Intent→Weft |
| `cnnl` / `intent:bangc` | CNNL / Intent→BANG C |
| `--baseline triton` | 从本次选中的实现中指定比较基准；不改变数值参考 |
| `--compiler /path/to/intent-compile` | 选择 Intent 编译器构建 |

用 `describe` 查看某个 workload 实际提供的实现。FlagGems 已从可运行 baseline 和数值参考中移除，相关 workload 改用 `torch`；旧 workload 名中的 `flaggems_` 仅用于保留原登记引用。

环境配置放在 `~/.config/intent-bench/environments/名称.json`，也可直接传 JSON 文件路径。CLI 显式参数覆盖配置；SDK、主线位置和机器只需配置一次。

```json
{
  "transport": "local",
  "python": "/path/to/cuda-venv/bin/python",
  "device": "cuda:0",
  "intent_root": "/path/to/intentdsl",
  "compiler": "/path/to/intent-compile",
  "threads": 8
}
```

```sh
intent-bench env list
intent-bench env show local-gpu
intent-bench run softmax --env local-gpu --case 64x1024-float16 \
  --impl torch,triton,cutile,intent:triton,intent:cutile --baseline triton
intent-bench run softmax --env local-gpu --case 64x1024-float16 --impl triton,intent:triton \
  --param 'rows=[1,64,512]' --param 'columns=[128,1024,4096]'
```

`examples/environments/` 提供本地 GPU、远端 GPU、MLU 配置示例。`transport: "ssh"` 的 `host` 使用用户已有的 SSH 别名，不保存密码或私钥。

执行层在 `execution.py`，只有这一层启动 SSH/SCP。普通 Python/GPU 算子在选定机器的 Python 中运行，传送 bench 包和单个自定义 workload 文件后在远端本地计时；自定义文件依赖的其他模块由目标 Python 环境提供。MLU/RVV 原生适配器使用同一执行层的持久 worker，支持本地和 SSH；Intent 编译可在控制端完成，native 编译与调用在目标机器完成。此时 `controller_python` 选择控制端 SDK 环境，`python` 选择 worker Python，`intent_root` 和 `compiler` 指向控制端文件。

SSH、文件传输和进程启动都在算子计时之外。实现适配器负责输入 ABI、SDK 调用和取结果；主线 compiler/runtime 不依赖 bench，也不负责 SSH。

## Agent 与插件

```sh
python -m pip install -e '.[mcp]'
intent-bench plugin --intent-root /path/to/intentdsl --output /tmp/intentdsl-plugin
```

命令生成可导入的插件目录和 ZIP，绑定当前 Python 环境。包含三个 MCP server：主线已有的 `intent_manual`、`intent_compiler`，以及新增的 `intent_bench`；手册和编译器功能直接调用主线实现。插件中的 skill 让 Agent 发现适用场景并知道如何选择实现、环境和参数。

bench MCP 提供 `list_workloads`、`describe_workload`、`list_environments`、`run_benchmark`、`run_status`、`compare_runs` 和 `check_mainline`。运行返回任务目录，Agent 用 `run_status` 取进度与结果，不需要重复启动。生成新候选与评测已有代码仍是两个独立动作。

例如对已加载插件的 Agent 说：“用 local-gpu 环境，比较 softmax 的 Triton、cuTile 和 Intent 两个后端，扫描 128/1024/4096 列，只给延迟。”Agent 会通过这些接口完成。支持 MCP 的其他客户端也可直接连接 `intent-bench-mcp`，不依赖 Codex UI。

## 自定义算子与实现

在普通 Python 文件中提供一个 `Workload`，无需修改 runner：

```python
import numpy as np
from intent_bench import Case, Workload, callable_implementation

bench = Workload(
    name="increment",
    cases=[Case("small", {"size": 1024})],
    inputs=lambda case, seed: (np.arange(case.parameters["size"], dtype="float32"),),
    implementations={
        "mine": callable_implementation("mine", lambda x: x + 1),
    },
    oracle=lambda case, args: args[0] + 1,
)
```

复杂实现使用 `Implementation(name, prepare)`。`prepare(case, inputs, environment)` 返回 `PreparedCall`，提供 `run`、`result`，以及需要的 `reset`、`compile`、`synchronize`、`close` 和 `native_measure`。`native_measure` 返回毫秒样本列表。多 kernel 和 host 编排组成一次完整 `run`。`examples/custom.py` 展示两个步骤和可变输入的恢复。

输入在各实现之间复制，保留 NumPy 共享存储视图；自定义对象可定义 `__deepcopy__`。正确性可以使用独立 callable、独立实现或自定义 `check(actual, expected)`。选择 baseline 不改变 oracle。

在已安装所选 SDK 的 Python 进程中，也可以直接调用同一个引擎：

```python
from intent_bench import benchmark, load_workload, load_environment

rows = benchmark(
    load_workload("softmax"),
    ["triton", "intent:triton"],
    environment=load_environment("local-gpu"),
    baseline="triton",
)
```

`benchmark` 接受当前进程的 Python 对象；需要切换整个 Python 环境或将普通 Python/GPU 算子送到远端时，使用 CLI 或 MCP 入口。原生 worker 的本地／SSH 选择在 Python API 中同样可用。

`examples/custom_softmax.py` 展示如何给现有 workload 添加自己的实现，同时比较两个原生 GPU baseline 和两个 Intent 后端。自定义实现包含多个 kernel、原地更新和测量范围外的状态恢复。

## 测量与结果

- 默认只显示 **延迟（median，毫秒）**；需要时加 `--throughput tflops` 或 `--throughput gbps`，只增加一个吞吐指标。没有工作量公式的算子不填造吞吐量。正确性和失败显示为状态。
- CUDA 默认用 event 测完整调用，每次恢复输入后填充至少 256 MiB 缓冲区清理 L2，缓存清理在 event 区间之外。参考 [TileLang profiler](https://github.com/tile-ai/tilelang/blob/main/tilelang/profiler/bench.py) 和 [Tritonbench do_bench](https://github.com/meta-pytorch/tritonbench/blob/main/tritonbench/components/do_bench/run.py)。`--cache warm` 显式选择热缓存；不会删除编译缓存。CPU/native 保留各自缓存策略并记录。
- 默认按约 25 ms 预热、100 ms 采样的预算估计次数（最多 10000 次）；`--samples` / `--warmup` 可覆盖次数。原始样本、编译/首次调用耗时和环境留在 `results.json` 供诊断；CSV 和 HTML 是简表。
- `compare` 显示前后延迟和变化百分比。只比较数值通过且设备、计时方式、范围和缓存策略一致的结果；不同策略不会混为一次性能变化。JSON 中保留 `speedup = baseline_ms / implementation_ms`。
- `--timing host` 包含 Python/C ABI 调用与完成同步；CUDA event 测量完整设备调用区间；CUDA Graph 是显式选项。原生计时在 native 进程内执行。
- 准备、编译、首次调用与热调用分开。输入恢复不计时；作者算法内必需的 workspace 初始化和多个 kernel 都计入完整调用。
- `--scope` 是调用合同，适配器必须实现所选范围，不能仅改标签。当前内置算子主要提供 `prepared_call`。
- 数值错误、输入问题、表达缺口、缺少实现、编译失败和环境缺失分别呈现。新增 bench 适配器尚未完成不等于 DSL 不可表达。
- `compare --max-regression 0.05` 可选择启用 5% 回归阈值；默认只展示变化。

## 内置资产与远端原生实现

`corpus/` 名称保留 309 个原始登记项的映射，历史输入规模和容差保留。`archive/` 保存旧报告与成绩，不能作为当前代码性能结论。新运行写入独立 `runs/`，不回写历史表。

```sh
intent-bench run --suite gpu --intent-root /path/to/intentdsl --compiler /path/to/intent-compile
intent-bench run corpus/mojo/relu --impl torch --threads 8
intent-bench run mlu/layer_norm --impl cnnl --option 'host="mlu-ssh-alias"'
intent-bench run rvv/q4_projection --impl ggml \
  --option 'host="rvv-ssh-alias"' \
  --option 'ggml_source="/path/to/llama.cpp"' \
  --option 'ggml_libraries="/path/to/llama.cpp/build/bin"'
```

远端实现使用用户已有 SSH 配置，传送输入并启动持久 worker，编译和传输均不在 native timer 内。GGML 需要目标机器上的 headers、共享库与 C++ 编译器；CNNL 默认使用 `/usr/local/neuware`。使用 `--option 'python="/path/to/python"'` 选择远端 Python；原生 worker 需要 NumPy，Intent worker 还需要 Python 3.10+。不会修改系统配置或管理凭据。

`mlu/{relu,softmax,matmul,layer_norm}` 支持 `--impl cnnl,intent:bangc`。`rvv/` 的五类 workload 保留 GGML baseline 和真实 Intent 表达，包括预打包 Q4_K×Q8_K dot 与独立 Q8_K 量化。完整 Q4 projection 在调用内量化 activation，weights 的 GGML 格式编码在准备阶段完成。`corpus/weft/` 的六个原有条目使用 Weft 自己的 artifact ABI，可仅选 `--impl weft`；`weft_root`、`weft_compiler` 配置源码与编译器，`deployment` 配置 ISA 参数，`--env` 选择执行机器。所选 Intent 构建缺少后端时会记录编译失败，baseline 仍可运行。

## Agent 工具

```sh
intent-bench agent evaluate --reference /path/to/TritonBench \
  --task abs --language intent --program /path/to/candidate.py \
  --intent-root /path/to/intentdsl --compiler /path/to/intent-compile --output runs/agent-abs
intent-bench agent generate --reference /path/to/TritonBench \
  --task abs --language triton --command '["my-agent", "{task_dir}", "{output}"]' \
  --output runs/new-submission
```

生成调用与评测独立，评测复用相同引擎和结果格式。外部生成命令由调用者管理；`--codex` 入口复用隔离状态的单次提交预设，需要外部 `--state-root`、`--intent-root`、`--triton-ref`。生成命令不自动获得 benchmark 反馈。普通自定义 workload 不继承 Agent 实验的 import 限制。

## 开发

```sh
python -m pip install -e '.[test]'
pytest
```

设备运行结果是单独证据。常规 CI 验证 API、状态/别名恢复、错误分类、结果比较、打包和无 SDK 的发现入口；真实 GPU/原生远端验证使用明确配置的设备环境。

[本次迁移验证与已知问题](docs/validation.md) 记录实际执行范围、失败归因和复现命令。

`device checks` workflow 仅手动触发，在带 `intent-bench` 标签、已配置 SDK 的 self-hosted runner 上执行。使用仓库变量 `INTENT_ROOT`、`INTENT_COMPILER` 配置主线，远端连接选项由调用者明确传入。

历史目录的原始数值不回写；`intent-bench report path/to/historical.csv --output /tmp/history.html` 可读取已知的旧表格式，转换微秒/毫秒并明确标为历史记录。`archive/migration.json` 列出旧文件的迁移归属，`archive/agent-candidates/sources.json` 列出引用的候选源文件。
