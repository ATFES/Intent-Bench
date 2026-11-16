# IntentDSL Bench

独立的算子正确性检查与性能比较工具。[English](README.en.md) · [第三方来源](THIRD_PARTY.md)

Intent 负责本机 JIT、目标探测、编译、ABI 绑定与执行；Bench 负责选择输入、实现、
数值对照、测量及 SSH 启动。作者算法和 30 个完整程序随 `intent.examples` 安装，
没有 Bench 时也能独立使用。

## 安装与选择

```sh
python -m pip install -e '.[mcp]'
export INTENT_BENCH_OUTPUT_ROOT=/path/to/intentdsl_bench/output
intent-bench list
intent-bench describe softmax
intent-bench env list
intent-bench run softmax --env local-gpu --case 64x1024-float16 \
  --impl intent:triton,intent:cutile,triton --baseline triton
intent-bench check --suite mainline --backend triton --env local-gpu
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
  数值通过的结果。数值失败的诊断时间不作为性能胜出。
- 纯 Python 或 eager CPU 实现如实报告 host 时间，不能冒充原生 CPU timer。

## Python API 与扩展

```python
from intent_bench import RunRequest, run, load_environment

outcome = run(RunRequest(
    "softmax", ("intent:triton", "triton"),
    environment=load_environment("local-gpu"),
    cases=("64x1024-float16",), baseline="triton",
))
```

自定义文件暴露 Workload，参见 [可变状态示例](examples/custom.py) 和
[GPU 多 kernel 示例](examples/custom_softmax.py)。
`Implementation.prepare` 返回 PreparedCall：launch、result，以及按需的
reset、compile、synchronize、close、native_measure。
参数绑定和输出分配在 prepare；编译在 compile；所有必需 kernel 在 launch。
设备 SDK 由被选择的实现延迟导入。

```sh
intent-bench run examples/custom.py:bench --impl mine
intent-bench compare output/daily/BEFORE output/latest/local-gpu/softmax
intent-bench report output/latest/local-gpu/softmax
```

## 实验与插件

`experiment paper --protocol protocol.json` 读取完整 RunRequest 列表
（顶层 name 和 requests）；每条必须显式给出 method、scope、cache、samples、warmup，
不继承日常默认协议，输出写入 experiments。原始论文表和 archive 保持不变。
Agent 的 generate/evaluate 分离；首次交付与不反馈测试结果给生成过程的规则不变：

```sh
intent-bench experiment agent evaluate --reference /path/to/TritonBench \
  --task abs --language intent --program /path/to/candidate.py \
  --output /path/to/intentdsl_bench/output/experiments/agent-abs/current
intent-bench plugin --output /path/to/intentdsl_bench/output/build/plugin
```

插件的 Intent 手册和编译 MCP 直接启动 Intent 包的入口；Bench 不生成手册副本。
Bench MCP 的 run_benchmark/check_mainline 提交同一请求服务，run_status 查询后台结果。
[插件安装](docs/codex-plugin.md) · [来源与许可证](THIRD_PARTY.md)
