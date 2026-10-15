# 迁移验证与已知问题

本文保留首次迁移时的运行记录。后续 FlagGems 已从活动 baseline 和数值参考中移除并替换为独立 Torch 实现；当前接口、环境配置和插件用法见 README，本文的历史结果不回写。

2026-10-15，本地 x86 / RTX 5090D，以及远端 SG2044、K1、MLU 实机验证。
结果默认只展示 median 延迟，吞吐量按需开启；本次设备存在其他任务，运行时间用于验证测量路径，不作为性能达标结论。

## 原有登记项

全部 309 项有独立适配器和迁移归属，均执行过。下表合并完整套件与适配器修正后的针对性复测，不将旧失败覆盖成成功，也不统计为多份用例。

| 范围 | 数值通过 | 当前其他结果 |
|---|---:|---|
| GPU 91 | 80 | 7 编译失败、1 reference 失败、2 上游 source 崩溃、1 超时 |
| x86 199 | 175 | 5 编译失败、17 超时、2 数值失败 |
| MLU 13 | 12 | 1 编译失败 |
| RVV 6 | 0 个 Intent 实现 | 所选 Intent 构建不提供 Weft；独立 Weft source 六项全部数值通过 |

CPU 全套采用 8 线程、4 个并行检查进程、单项 240 秒超时；GPU 检查串行。超时保留原生产输入，不能解释成不支持该算子。普通性能测量始终串行。

## 新 baseline 与接口

- 九个独立 Mojo baseline：ReLU、SwiGLU、transpose、embedding、jagged mean、prefix sum、CSR SpMV、histogram、AdamW；两个规模共 18 个数值检查通过。
- 十一个原 `run_only` 算子：独立参考和实际 Mojo 表达均已接入；small/medium 共 22 项，以及全部 11 个原生产规模的 Mojo 数值检查通过。对应 Intent small/medium 检查 20/22 通过，chunk retention 两项为编译失败。
- GGML：FP32 GEMM、RMSNorm、Q4_K projection 的 decode/prefill、Q4_K×Q8_K dot、Q8_K quantization 均在 SG2044 通过。量化输入按实际 GGML 格式处理。
- CNNL 与 Intent→BANG C：ReLU、Softmax、Matmul、LayerNorm，两个规模、两种实现共 16 项在同一 MLU 上通过。
- 新的 Q4 dot、Q4 projection、Q8 quantization Intent 表达已生成代码并通过 Mojo 数值验证；这不代表已经验证 Weft 运行。所选 Intent 构建缺 Weft 的失败完整保留。
- 五方 softmax（Torch、原生 Triton、原生 cuTile、Intent 两后端）通过 CUDA event 测量；独立文件可扩展实现。`examples/custom_softmax.py` 另提供带输入恢复的多 kernel 实现。
- 干净 wheel 安装后，在没有 Intent/Torch SDK 的虚拟环境中发现 340 个 workload、自定义状态恢复示例和独立 Mojo AdamW 均通过；GPU 环境中的安装包完成用户实现加上述五种实现的六方数值与计时验证。HTML 已实际渲染检查。
- 主线 30 个作者程序均执行成功，记录为 `executed`，不冒充数值校验。Agent 的真实候选重新评测和 mock 外部生成入口分别通过；重新评测不调用模型。

## 需要主线或环境后续处理的问题

### FP8 输出舍入边界

`corpus/mojo/fp8_gemm` 保持原 `atol=0.5, rtol=0.05`，仍记录数值失败。
seed 0、8 线程下，输出 `(15,113)` 的 float64 点积为 `116.00000381469727`。
完整 Torch GEMM 累加值为 `116.00000762939453`，转 FP8 得 `120`；抽出同一行和列做小 GEMM 得 `116.0`，转 FP8 得 `112`，与 Intent 输出相同。
这证明该失败对累加和最终 FP8 舍入边界敏感，尚不能仅据此归因于编译器错误；没有放宽容差消除失败。

### RoPE 缓存更新

`corpus/mojo/padded_rope_cache_update` 出现非有限值和未正确写入的输出。
生成的 Mojo 源码把共享的 `half_indices` 临时存储加 64 后写回，后续 head 再次读取并修改该存储；这与定义中反复使用原索引的语义不符。
输出用 NaN 初始化后的诊断确认部分预期位置仍非有限。对应源码和 IR 已保留；bench 不修改主线 lowering。

### 其他失败

GPU 中 FlagGems 的 transpose、addcmul 在 source-only 运行也会于 `pointwise_dynamic → libentry → Triton launcher` 崩溃，发生在 Intent 编译之前。FlashAttention reference 的 `PassManager::run failed` 单独标记。cuTile recurrent gated delta 保留超时结果。

其余编译失败包括 GPU 的 MoE、scaled FP8 split-K、fused add RMSNorm、batch norm、grouped decode，CPU 的 block sparse、gated delta、retention、带 sink logits 的变长 GQA，以及 MLU causal attention。具体错误、输入标识、命令和已有产物位置在结果 JSON 与 worker 日志中。

## 复现

```sh
export INTENT_MOJO=/path/to/mojo
export INTENT_BENCH_MOJO=/path/to/mojo
intent-bench run --suite cpu --check-only --jobs 4 --threads 8 --timeout 240 \
  --intent-root /path/to/intentdsl --compiler /path/to/intent-compile --output runs/cpu
intent-bench run corpus/mojo/fp8_gemm --impl intent:mojo --check-only --threads 8 \
  --intent-root /path/to/intentdsl --compiler /path/to/intent-compile --output runs/fp8
intent-bench run corpus/mojo/padded_rope_cache_update --impl intent:mojo --check-only --threads 8 \
  --intent-root /path/to/intentdsl --compiler /path/to/intent-compile --output runs/rope
intent-bench run examples/custom_softmax.py:bench --case 64x1024-float16 \
  --impl mine,triton,cutile,intent:triton,intent:cutile --device cuda:0 \
  --intent-root /path/to/intentdsl --compiler /path/to/intent-compile --output runs/extension
```

本次原始运行与合并索引保存在执行机器的 `~/.cache/intent-bench/validation/`，不提交设备产物。`current/results.json` 按登记项保存当前 309 个 Intent 检查结果，并用 `metadata.validation_source` 指向实际运行文件；`current/manifest.json` 说明针对性复测与补充验证的来源。

## 文件迁移

原 618 个非缓存文件全部登记在 `archive/migration.json`：516 个直接逐字节一致，102 个重构后迁移。历史表的 784 个候选引用都有归属，外部候选只提取源文件及相邻结果；配置、认证、模型状态和日志未整目录搬入。

主线原 `experiments/`、`report/` 已迁出；恢复副本保存在执行机器的 `~/.cache/intent-bench/migration-backup/20261015/`。工具安装包不包含历史目录、旧成对 runner 和旧 provider 树。主线只通过可选 CLI 调用 bench，编译器不依赖 bench。
