# 03 · mini-pi —— Python 极简 Coding Agent + 轻量自进化

**目标：自己写一个能改代码的 Agent 内核**，不是调别人的框架。

它和项目 02 的区别：02 是「用工具回答问题」，03 是「**动磁盘、跑命令、改代码**」——
所以多了三样东西：**轨迹（trace）**、**回归集（cases）**、**评测（evals）**。

---

## 一、版图：谁调谁

```
loop.py ──→ core/llm.py ──→ DeepSeek API
   │
   ├──→ core/tools.py ──→ 文件系统 / 子进程
   └──←── 结果字符串

实际入口只有两条：
  python loop.py            loop.py 的 __main__ REPL，交互式
  python -m core.evals      批量评测，直接调 loop.run_agent()
没有独立的入口层文件。原先占位的 内核.py 已于 2026-11-13 删除（git 历史可恢复）。
```

| 模块 | 职责 |
|---|---|
| `loop.py` | **当前实际入口 + 驱动**（在根目录，不在 `core/`）：多轮闭环、两个终止条件、REPL；`run_agent()` 被评测层直接调用 |
| `core/llm.py` | **全项目唯一的模型出入口** |
| `core/tools.py` | **唯一会改动磁盘或启动子进程的模块** |
| `core/trace.py` | 每轮工具调用落盘（jsonl），可回溯 |
| `core/dataset.py` | 评测数据集抽象：把「跑什么任务」和「怎么判分」解耦 |
| `core/checks.py` | **断言注册表**：每类断言一个 `@register` 函数，新增类型不动主链 |
| `core/evals.py` | **评测中台统一入口**：跑用例 / 落报告 / 前后对比 / 自检 / **按报告回放证据** |
| `tests/test_checks.py` | 零依赖回归测试（标准库 `unittest`，不联网、不需要 key） |
| `reports/` | 每次评测落一份 JSON 报告，供 `--compare` 前后对比、`--evidence` 回放到轨迹 |
| `check_llm.py` | 初始验证存档，不参与运行时 |

**模块之间只传三样东西**：
1. `messages` 列表（只增不减）
2. `name + arguments_json`（循环 → 工具）
3. 工具返回值（字符串，工具 → 循环）

⭐ 这三条边界是刻意压的：**传的东西越少，后面加功能越不容易乱。**

**接线后的写操作链路（2026-11-12 新增）**：

```
ask → tool_calls → [Skill 已注入 system 消息]
   → core/approval.propose()      出确认卡，落 action_card 事件
   → core/approval.check_gate()   复核工具名 + 参数指纹；拦下落 blocked 事件
   → run_tool()                  才真正动磁盘
   → core/approval.make_receipt() 落 receipt 事件（含实测落盘字节数）
   → 回填 role="tool"             门拦下也回填，否则下一轮 invoke 报 400
```
四类新事件都由 `core/trace.py:90 write(**event)` 直接写入 —— 写入层不做白名单校验，
所以 `trace.py` 一行没改。评测层从 JSONL 读回这些事件做断言，不读模型的 `messages`。

---

## 二、两个终止条件（不是"跑满就停"）

1. **正常退出**：模型这一轮没返回 `tool_calls` —— 它认为信息够了，直接作答
2. **强制退出**：轮数到达 `MAX_TURNS`（默认 20）—— 防止死循环烧钱

⭐ 只有一个终止条件的 Agent 是有问题的：任务答完了还会被判"超限"。
两个出口才能区分**"它就是答完了"** 和 **"它是被截断的"**。

---

## 三、为什么"Coding Agent"必须有 trace 和回归集

普通问答 Agent 错了就错了；**Coding Agent 错了是改坏你的文件。**

| 机制 | 解决什么 |
|---|---|
| `trace/*.jsonl` | 「你说它跑通了，凭什么」—— 每次工具调用都有记录 |
| `cases/` | 「改了之后有没有变差」—— 同一批任务，前后对比 |
| `core/evals.py` | 把上面的对比自动化（统一入口） |
| `core/checks.py` | 「适配度高」的机制本体 —— 断言可注册、可插拔 |
| `reports/` + `--compare` | 「改了能证明没改坏」—— 机器判据，退化就非零退出 |
| `--evidence` | 「这个 PASS 凭什么」—— 报告每轮记轨迹相对路径 + 断言明细，缺证据就非零退出 |

现有 3 个回归用例，各带 fixture：`cases/read_then_write/`（读 → 写 → 断言内容）、
`cases/copy_by_command/`（必须走 `run`，禁调 `write_file`）、`cases/long_output/`（长文本 + 回读）。
三条都写了 `expect.receipt` 与 `expect.skill_injected`，即除了文件内容，
还要核验确认卡、回执与技能注入这条证据链。

**评测中台**：断言不再写死在 `check()` 里。新增一类断言 = 在 `core/checks.py`
写一个函数 + 一个 `@register`（装饰器），`evals.py` 的 `check()` 一行都不用改 ——
本轮新增的 `skill:injected` 就是这么挂上去的。所有断言都带「故意改坏必须报警」的自检
（`--selfcheck`，10 项），回归测试在 `tests/`（11 项，标准库 `unittest`）。

**报告 → 证据回放**（本轮补的断链）：`build_report()` 每轮记 `evidence` ——
轨迹**相对**路径（不写本机绝对路径，远端是公开仓库）+ 该轮断言明细。
`--evidence` 按报告找出每轮轨迹、打印事件序列，并校验文件存在、非空、有终止事件；
旧报告、空报告、被删轨迹、被手改结论一律非零退出。

---

## 四、怎么跑

```bash
pip install openai python-dotenv

# 交互 REPL：输入任务后回车；写操作会弹确认卡，输入 y 才执行
# （loop.py 的 __main__ 是 REPL，不接收命令行任务参数）
python loop.py

# 评测中台：跑全套（每条 3 轮，pass^3 才算过）+ 报告落盘 reports/
python -m core.evals

# 断言层自检：故意改坏 10 处，断言必须报警；不联网、不需要 key
python -m core.evals --selfcheck

# 两份报告前后对比，发现退化就非零退出（CI 可直接接）
python -m core.evals --compare reports/A.json reports/B.json

# 按报告回放证据：打印每轮轨迹路径与事件序列，缺证据就非零退出
python -m core.evals --evidence reports/A.json

# 零依赖回归测试（标准库 unittest，不需要安装任何第三方包）
python -m unittest discover -s tests -v
```

`trace/` 下是历史运行轨迹，可以直接打开看每一步。真实内容长这样（录自 `trace/2026-11-07_111315_389228.jsonl`；**第 2 行 `content` 字段已截断，完整内容见源文件**）：

```jsonl
{"type": "tool_call", "turn": 1, "name": "read_file", "args": "{\"path\": \"a.txt\"}", "md5": "0a8c1b0d5dfb", "len": 34, "ts": "11:13:16"}
{"type": "tool_call", "turn": 2, "name": "write_file", "args": "{\"path\": \"b.txt\", \"content\": \"...\"}", "md5": "bd3ddf1daf5a", "len": 16, "ts": "11:13:17"}
{"type": "final", "turn": 4, "chars": 96, "ts": "11:13:18"}
```

字段含义：
- `name` 是工具名（不是 `tool`）；`args` 是**原始字符串**（不是对象）
- `md5` / `len` 是参数指纹与长度 —— 用来判断「同样的调用有没有重复」
- 末尾 `type: "final"` 是正常终止标记（没有它说明这次是被截断的）

---

## 五、当前进度与已知缺陷（如实写）

| 项 | 状态 |
|---|---|
| 单轮闭环 | ✅ |
| 多轮闭环 + 双终止条件 | ✅ |
| 工具层（读/写/跑） | ✅ |
| trace 落盘 | ✅ |
| case 回归集 + 评测 | ✅（3 个用例，含 fixture；每条跑 3 轮） |
| 确认卡 / 权限门 / 回执 | ✅ `core/approval.py`：写工具先出卡、批准后校验指纹，执行后落 `type="receipt"` |
| Skill 按需加载 | ✅ `core/skills.py`：命中触发词才注入一条 system 消息，不进 `TOOLS` |
| 回执类断言 | ✅ `core/checks.py`：exists / source / fingerprint / executed / approved_by 五条 |
| **评测中台** | ✅ 断言注册表 + 统一入口 + 报告落盘 + `--compare` 前后对比 + `--selfcheck` 反向验证 + **`--evidence` 证据回放**；`tests/` 11 项零依赖回归 |
| 工具序列断言 | ✅ `case.expect.tools_must` / `tools_forbidden`：不只判结果，还判「该调的工具调了没、不该调的有没有调」 |
| pass^k（可靠性） | ✅ 每条用例跑 3 轮（`RUNS = 3`），**全过才算过**；报告同时给单次通过率与 pass^3 |
| **自进化模块** | ⬜ **未完成** —— 计划：把参数化配置 + 评测线接起来，自动搜索更优配置 |
| 安全 | ⚠️ **评测有沙箱，裸跑没有**：`core/evals.py` 每条用例建独立临时目录（`%TEMP%\mini_pi_*`）+ `os.chdir` + `finally` 销毁，**异常路径也销毁**；但 `python loop.py` 裸跑是直接操作当前目录。**写操作现在要过确认卡**（裸跑传 `approver="human"`，评测传 `"eval"` 并标记 `eval:auto`），但确认卡管的是**授权**，不是路径白名单或危险命令拦截 —— 这两样仍然没有 |

⭐ **最后两条是最该被追问的地方，也是最诚实的部分。**

---

## 六、自进化模块的计划（还没做，先说清要做什么）

不是"让 LLM 改代码"（不可控、难复现），而是：

```
① 把可变部分参数化（最大轮数 / 工具重试次数 / prompt 模板）
② 每代只改 1–2 个参数
③ 用 cases/ 跑回归打分
④ 三条判据同时满足才采纳：通过率不降 + 无用例退化 + 可复现
⑤ 记进化日志（第几代 / 改了什么 / 分数）
```

⭐ **瓶颈不在搜索空间，在评测集大小**：只有 3 个 case 时，"更优"很可能是在拟合噪声。
所以先扩 cases，再做自进化 —— 顺序不能反。
