# Codex 插件

插件名为 `intentdsl`，界面名称为 **IntentDSL Bench**。包含 bench、Intent
语言手册和编译器 MCP，以及 Agent 选择比较对象、输入和环境的 skill。
源码位于 `src/intent_bench/plugin/`，随 Python wheel 分发。

## 安装

先在一个长期保留的 Python 环境中安装包和 MCP 依赖，再导出插件：

```sh
python -m pip install '.[mcp]'
python -m pip install 'intentdsl[manual]'
intent-bench plugin --output "$HOME/.codex/plugins/local/intentdsl"
```

导出目录必须尚不存在。导出器把该 Python 的绝对路径写入 MCP 配置，
不依赖 Codex 进程的 shell PATH。运行 GPU/原生算子的 SDK 环境可另外通过
bench 环境配置选择；插件控制端安装 Bench 的 MCP extra 和 Intent 的 manual extra。
手册、编译器服务直接启动 Intent 自己的入口，不需要源码路径或临时手册副本。

在个人市场 `~/.agents/plugins/marketplace.json` 的 `plugins` 数组中加入：

```json
{
  "name": "intentdsl",
  "source": {"source": "local", "path": "./.codex/plugins/local/intentdsl"},
  "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
  "category": "Developer Tools"
}
```

保留市场原有名称和其他插件；上述路径相对于个人市场根目录（用户 home）。
使用支持插件命令的 Codex CLI 安装，其中 `YOUR_MARKETPLACE` 是市场顶层 `name`：

```sh
codex plugin add intentdsl@YOUR_MARKETPLACE
codex plugin list --marketplace YOUR_MARKETPLACE --json
```

也可以从桌面插件目录选择该本地市场安装。进入新对话后使用插件；如果当前
桌面的插件列表仍显示旧状态，重新打开应用以刷新。
格式与市场机制参见 [OpenAI 插件文档](https://developers.openai.com/plugins/build/plugins)。

## 对 Agent 说什么

- “用 IntentDSL Bench，把 softmax 的两个 Intent GPU 后端和一个 Triton baseline 比一下。”
- “保持实现不变，扫描这几个输入规模，只给延时。”
- “把我这个 Python 文件里的算子接入 bench，使用已有输入和 baseline。”
- “比较这两个已保存的运行结果。”

Agent 查询实际用例和环境，明确指定参与列表。两个 Intent 后端可以共用一个
baseline；baseline 必须包含在参与列表中。选实现与选 SSH/本地环境分别处理。
默认输出 median 延时，需要时再加一项吞吐量。
普通运行默认测准备好的算子执行：GPU 图重放、图内 event 计时，输入恢复和
清缓存在图外。只有明确要求分析 Python/API/framework 开销时，Agent 才设置
`scope="host_call"`；不会在日常比较中额外运行或展示 Python 包装成绩。

手工使用同一个比较入口：

```sh
intent-bench run softmax --env local-gpu --case 64x1024-float16 \
  --impl intent:triton,intent:cutile,triton --baseline triton
```

`local-gpu` 是示例环境名，应从 `intent-bench env list` 选择本机实际配置。
MCP 的 `run_benchmark` 返回后台任务目录，`run_status` 读取结果，避免重复运行。

## 更新

更新插件所用 Python 环境中的 bench 包，导出到新的插件目录，修改市场中
此插件的 `source.path` 后再次执行 `codex plugin add intentdsl@YOUR_MARKETPLACE`。
Codex 使用安装缓存中的 skill 和配置，单独修改仓库源码不会更新已安装的副本。
配置中不包含凭据，机器连接沿用环境配置和用户自己的 SSH 设置。
