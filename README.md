# IntentDSL Bench

0.3.0 版在三轮完整调用的离散度之外判断快慢。IntentDSL 2.1.0 里程碑有 336 项同设备对照，244 快、85 平、7 慢。[计分板](https://github.com/Tianchen-ai/Intent/blob/main/doc/plan/v0.6-semantics.md)

独立的算子正确性检查与性能比较工具。[English](README.en.md) · [第三方来源](THIRD_PARTY.md)

Intent 负责本机 JIT、目标探测、编译、ABI 绑定与执行；Bench 负责选择输入、实现、
数值对照、测量及 SSH 启动。V2 的作者程序随 `intentdsl.examples` 安装，`intent:*` 选择 V1，
没有 Bench 时也能独立使用。

## 安装与选择

在本仓根目录执行，沿用 [IntentDSL 源码安装](https://github.com/Tianchen-ai/Intent#build-from-source) 的 GPU 环境。

```sh
python -m pip install -e .
export INTENT_BENCH_OUTPUT_ROOT="$PWD/output"
intent-bench describe softmax
intent-bench run softmax --env examples/environments/gpu.json --output-root "$PWD/output" --case 64x1024-float16 --impl intent-v2:triton,triton --baseline triton --check-only
intent-bench run softmax --env examples/environments/gpu.json --output-root "$PWD/output" --case 64x1024-float16 --impl intent-v2:triton,triton --baseline triton
```

也可在 `~/.config/intent-bench/config.json` 设置 `{"output_root":"/absolute/path/output"}`。
`--impl` 必填，baseline 必须是本次显式选择的实现之一；它与数值 oracle 独立。
同一原生 baseline 可以与多个 Intent 后端比较。Torch 只是按需选择的独立实现，
普通调用不会自动加上它或额外框架包装。

| 入口 | 用途 |
|---|---|
| list / describe | 查询 workload、实现、case、环境及来源；不编译、不启动设备 |
| run | 正确性与完整算子计时 |
| check | 安装包中的作者 program，分别检查 source/native/run |
| experiment paper / agent | 显式论文协议、Agent 生成或已有候选评测 |
| compare / report | 读取和展示已有结果 |

`check` 的 executed 只表示执行完成；数值对照使用 run。
既有 workload 标识、输入形状、dtype 和容差保留；FlagGems 历史名称不表示活跃的
FlagGems baseline。使用 describe 查询当前可选实现。

## 执行环境

环境位于 `~/.config/intent-bench/environments/NAME.json`，也可直接传配置路径。
可复制 `examples/environments/remote-gpu.json` 或 `mlu.json` 填写自己的 SSH 别名与工具链路径。仓内私有配置可放在 `local-config/`，用 `--env local-config/NAME.json` 读取。
复现归档 Weft 实验时，将 `archive/legacy-tools/cpu/providers/weft/{ime,rvv}.json` 模板复制到 `~/.config/intent-bench/legacy-weft/` 并填写目标配置，也可用 `INTENT_WEFT_PROFILE` 指定文件。
同一份 RunRequest 被 CLI、Python API、MCP 和 suite 使用：

```json
{
  "transport": "ssh",
  "host": "gpu-server",
  "python": "/path/to/venv/bin/python",
  "remote_output_root": "/path/to/intentdsl_bench/output",
  "device": "cuda:0",
  "threads": 8
}
```

选定解释器必须已经安装 Bench、Intent（若选择 Intent）及所需 provider。
SSH 只传输请求和显式的用户 workload 文件，普通运行不会复制产品源码或把控制机
生成的 artifact 当作远端 JIT。compiler 路径若指定，也是执行机上的路径。
`intent_root` 仅检查已安装的 editable checkout，不修改导入路径。
CPU 的 `cpus` 选择允许的执行集合，`threads` 是线程预算，ISA/VLEN 由产品探测。
MLU 使用 `device: "mlu:0"`，NPU 使用 `device: "npu:0"`。Intent provider 选项按原样放在
`options.intent.<backend>`，例如 `{"intent":{"mojo":{"compiler":"/path/to/mojo"}}}`
或 `{"intent":{"bangc":{"neuware":"/usr/local/neuware"}}}`。新增 provider 不需要修改 Bench 的工具链映射。
`check --backend` 使用安装的 Intent 组件登记；程序比较也公开这些组件，不以是否已有同设备 baseline 限制正确性检查。
缺少设备计时适配器时明确报错，不以 Python 提交时间冒充算子时间。

执行机缺少原 oracle 的依赖时，可显式选择 `--oracle-env local-cpu`。
Bench 在执行机完成本机 JIT 和测量，回传实际输入、初始状态与输出；所选环境运行原
oracle 和原检查器。传输保留 dtype、stride 和共享存储关系，不重新按 seed 猜测输入。
这不会把 oracle 环境变成性能 baseline；性能对照仍必须在同一执行设备上计时。
Python 的 `RunRequest.oracle_environment` 与 MCP 的同名参数使用相同服务。

异机验数时，给 `run --check-only` 传入目标环境文件和 oracle 环境文件。

`options.triton_ptxas` 和 `options.triton_ptxas_blackwell` 分别传给 Triton 的
`TRITON_PTXAS_PATH` 与 `TRITON_PTXAS_BLACKWELL_PATH`，路径属于执行主机。
它们显式选择该 worker 中所有 Triton 实现使用的汇编器，不在失败后切换工具链。
GGML/CNNL 等独立 baseline 的 library/header 路径仍由各自环境配置。

超时按请求处理；失败保留阶段和日志，不自动换后端或计时方式。
同一设备的 Bench 调用互斥，重叠请求报告 skipped_busy。
外部负载与设备占用仍应在测量前检查。

## 输出

```text
output/
  latest/<environment>/<workload>/
  daily/YYYY-MM-DD/<environment>/
  experiments/<experiment>/<run>/
  cache/<environment>/
  build/
  work/
```

本轮结果替换对应 latest 中的结果，即使失败也不展示上一轮成功。
编译产物与共享缓存独立保留；worker 暂存由执行服务回收。
服务统一设置 Intent、provider、XDG 与临时目录，远端采用其 remote_output_root。
历史 archive 保留，不被当前结果覆盖。

## 测量合同

- 默认 scope=operator。GPU 使用完整算子的图重放，timing events 在图内，
  全部 kernel 和算法必需的状态更新都在区间内；输入恢复、L2 清理与 Python 提交在外。
- CUDA 默认清理至少 256 MiB，或大于设备 L2 的缓存；cache=warm 显式选择热缓存。
  图捕获失败会报错，可显式选择 timing=cuda_event。
- CPU 原生实现提前分配和绑定，多 kernel 在一个 C ABI 原生计时间隔内执行。
  计时器使用 CXX 或 c++ 构建。MLU 使用完整队列的 CNRT notifier 区间。
- host_call 仅在显式选择时构造和测量 Python/框架入口。普通结果不包含额外包装成绩。
- 编译、首次调用/调优、热调用分别记录。默认按 25 ms 预热和 100 ms 采样预算估计次数，
  samples/warmup 可显式覆盖。原始毫秒样本保存在 JSON；CSV/HTML 是简表。
- speedup=baseline_ms/implementation_ms，只比较设备、scope、计时边界和缓存策略匹配且
  数值通过的结果。每侧至少有 3 个样本时，用候选与参照的中间 50% 样本计算倍率范围 `[Q1候选/Q3参照, Q3候选/Q1参照]`。整个范围低于 0.95 或高于 1.05 才判快或慢，其余判平。JSON 和 CSV 保存范围与四分位距（IQR）。多轮同口径采样可传给 `intent_bench.comparison.compare_latency`，各轮方向一致且超出阈值才判快慢。
- 纯 Python 或 eager CPU 实现如实报告 host 时间，不能冒充原生 CPU timer。

## Python API 与扩展

```python
from intent_bench import RunRequest, run, load_environment

outcome = run(RunRequest(
    "softmax", ("intent-v2:triton", "triton"),
    environment=load_environment("examples/environments/gpu.json"),
    cases=("64x1024-float16",), baseline="triton",
))
```

自定义文件暴露 Workload，参见 [可变状态示例](examples/custom.py) 和
[GPU 多 kernel 示例](examples/custom_softmax.py)。
`Implementation.prepare` 返回 PreparedCall：launch、result，以及按需的
reset、compile、synchronize、close、native_measure。
参数绑定和输出分配在 prepare；编译在 compile；所有必需 kernel 在 launch。
设备 SDK 由被选择的实现延迟导入。

对比已有结果使用 `compare`，重建报告使用 `report`。

## 实验与插件

`experiment paper --protocol protocol.json` 读取完整 RunRequest 列表
（顶层 name 和 requests）；每条必须显式给出 method、scope、cache、samples、warmup，
不继承日常默认协议，输出写入 experiments。原始论文表和 archive 保持不变。
Agent 的 generate/evaluate 分离；首次交付与不反馈测试结果给生成过程的规则不变：

实验命令的参数和输入格式可通过 `intent-bench experiment --help` 查询。

插件的 Intent 手册和编译 MCP 直接启动 Intent 包的入口；Bench 不生成手册副本。
Bench MCP 的 run_benchmark/check_mainline 提交同一请求服务，run_status 查询后台结果。
[插件安装](docs/codex-plugin.md) · [来源与许可证](THIRD_PARTY.md)
