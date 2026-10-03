# ai-agent-learning
> 个人AI学习实战仓库 | 个人学习记录。从 Python 基础补起，逐步手写实现 AI Agent 与 RAG 的核心链路。
> 
> 说明：仓库中的早期代码有一部分是在 AI 辅助下完成的，我正在逐个重写成能独立讲解、可面试复盘的版本。本文档仅展示目前已完全吃透、能讲清原理的内容。

## 📁 项目模块
- `simple_agent_demo.py`：简易对话客户端，实现对话记忆持久化（JSON）、历史截断、异常处理（早期练习作品，AI辅助比例较高，尚未重写；暂不纳入「项目深度复盘」范畴，不作为面试展示项目）
- `basic_rag_pipeline.py`：内存版完整 RAG 链路，纯手写分块、向量化、检索、Prompt 组装（已吃透、全量自测）
  - 💡 关键设计决策（原创优化）：`basic_rag_pipeline` 递归切分 overlap 防重复叠加设计
    - 传统递归分块存在缺陷：每一层递归都做 overlap 更叠上下文，递归越深，重叠内容叠加越多，最终文本大量冗余、失真。
    - 我增设 **_raw 标记参数**，做层级控制：
      - ✅ 只在最外层做一次 overlap 拼接 + 参数校验
      - ✅ 内层递归只负责切分文本，不再重复叠加重叠区域，大幅减少冗余文本，同时避免重复参数校验，提升分块精度与执行效率。
- `tool_agent.py`：无框架手写 ReAct Agent（DeepSeek API + Function Calling），完整实现工具调用循环、分层重试、边界容错
- `tool_agent_langgraph.py`：LangGraph `StateGraph` 版 ReAct Agent，复用 `tool_agent.py` 的工具定义/映射/重试，改用状态图（`add_messages` 自动归并消息 + 条件边路由）实现推理循环，并挂 `SqliteSaver` checkpointer 把状态落盘，支持跨轮对话记忆
- `practice/`：日常练习归档目录，按「专题_序号_名称」命名，不再散落在 PyCharm 工程里（当前 LangGraph 四练：`01` TypedDict 基础 → `02` 状态手动流转 → `03` StateGraph + add_messages → `03_Annotated` 在 03 基础上加 SqliteSaver，验证状态持久化）

### 📦 项目依赖
- `sentence-transformers>=2.7.0`
- `numpy>=1.26.0`
- `tiktoken`
- `requests`
- `python-dotenv`
- `langchain-openai`
- `langgraph`

---

## 🚀 Quick Start

### 1. 获取代码
```bash
git clone https://github.com/luotianwen123/ai-agent-learning.git
cd ai-agent-learning
```

### 2. 环境准备
建议 Python 3.9+。仓库未提供 `requirements.txt`，需手动安装以下依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install "sentence-transformers>=2.7.0" "numpy>=1.26.0" tiktoken requests python-dotenv langchain-openai langgraph
```

### 3. 配置 API Key
四个模块均调用 DeepSeek API（模型 `deepseek-chat`）。在项目根目录新建 `.env` 文件：

```dotenv
OPENAI_API_KEY=你的_DeepSeek_API_Key
# 可选：国内直连 HuggingFace 缓慢时走镜像（basic_rag_pipeline.py 首次运行要下模型）
HF_ENDPOINT=https://hf-mirror.com
```

> `.env`、`config.json`、`history.json` 均已在 `.gitignore` 中，不会被提交。

### 4. 运行各模块

**① `tool_agent.py` —— 手写 ReAct Agent（推荐先跑这个）**

```bash
# 直接提问
python tool_agent.py "现在几点了？帮我算一下 (128 + 370) * 3"

# 自定义最大推理轮次（默认 5）
python tool_agent.py "北京现在天气怎么样？" --max-steps 8

# 不传问题则打印帮助
python tool_agent.py
```

内置 4 类工具：时间查询、计算器、本地文件读取、真实天气 API 请求。
运行后会依次打印模型的思考、每次工具调用与回填结果，最后输出 `===== 最终答案 =====`。

**② `basic_rag_pipeline.py` —— 手写完整 RAG 链路**

```bash
python basic_rag_pipeline.py
```

⚠️ 四点注意：
- 首次运行会自动下载 embedding 模型 `BAAI/bge-small-zh-v1.5`（约 100MB），需要联网等待。国内网络慢时，在 `.env` 里配 `HF_ENDPOINT=https://hf-mirror.com` 走镜像。
- **`.env` 必须在所有第三方库 import 之前加载**：`huggingface_hub` 在模块导入时就会读取 `HF_ENDPOINT` 并存入常量，之后才调 `load_dotenv()` 已经太晚，镜像不会生效。调整 import 顺序时别把这个先后关系弄反。
- 当前为演示脚本，待检索的文档（`demo_doc`）和用户问题（`query`）写死在 `__main__` 中。想换内容直接改这两处变量即可。
- 脚本会打印组装完成的完整 Prompt，并调用大模型输出最终回答。

**③ `simple_agent_demo.py` —— 带持久化记忆的对话客户端**

该模块从 `config.json` 读取配置（文件被 gitignore，需手动创建）：

```json
{
  "agent_name": "学习助手",
  "system_prompt": "你是一个耐心的 AI 学习助手。",
  "api_key": "你的_DeepSeek_API_Key",
  "api_url": "https://api.deepseek.com/chat/completions",
  "max_history_len": 10
}
```

然后运行：

```bash
python simple_agent_demo.py
```

- `max_history_len` 可选，默认 10，超出后自动截断最早的历史。
- 对话中每轮都会把历史写入 `history.json`，下次启动可延续上下文。
- 输入 `exit` 退出。

**④ `tool_agent_langgraph.py` —— LangGraph 版 ReAct Agent**

```bash
python tool_agent_langgraph.py
```

复用 `tool_agent.py` 的全部工具定义、工具映射、重试装饰器与最大步数配置，改用 LangGraph `StateGraph` 搭建推理循环：`messages` 用 `Annotated[list, add_messages]` 让框架自动追加、`route_agent` 条件边判断「继续调工具 / 终止」，再挂 `SqliteSaver` checkpointer（状态落在脚本同目录的 `checkpoints.sqlite`，路径用 `os.path.dirname(os.path.abspath(__file__))` 锚定、不受启动目录影响）、靠同一个 `thread_id` 让多次 `invoke` 共享记忆。需额外安装 `langchain-openai`、`langgraph`。

### 5. 常见问题
| 现象 | 原因与处理 |
| --- | --- |
| `OPENAI_API_KEY 未配置，请在 .env 文件中填写` | `.env` 缺失或变量名为 `OPENAI_API_KEY` 拼写有误 |
| `配置文件不存在: config.json` | `simple_agent_demo.py` 未创建 `config.json`，见上方步骤 ③ |
| 首次运行卡住 / 下载缓慢 | 正在拉取 embedding 模型。直连 HuggingFace 慢时在 `.env` 配置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像；注意 `load_dotenv()` 必须排在第三方库 import 之前才会生效 |
| 已配 `HF_ENDPOINT` 但仍在直连 HuggingFace | import 顺序反了，见运行步骤 ② 的第 2 条注意 |
| 调用报网络错误 | 脚本对临时性网络错误做了延迟重试；持续失败请检查 API Key 余额与网络代理 |

---

## ✨ 项目深度复盘（简历 / 面试 完整版）
> 更新时间：2026-10-02 | 素材来源：真实 Git 提交记录 + 本地版本比对，无虚构、可核验

### 项目一：`tool_agent.py` | 无框架手写 ReAct Agent
1. **实现内容**
基于 DeepSeek API + Function Calling，零框架手写 ReAct 智能体。核心采用 while 循环实现模型思考、工具调用、结果回填的完整闭环。通过「工具名 - 函数对象」映射表，实现动态查表调用工具，直至模型主动结束任务。代码规模约 190 行，内置 4 类可用工具：时间查询、计算器、本地文件读取、真实天气 API 请求，支持命令行直接调用与帮助文档提示。

2. **核心问题与解决思路**
   - **问题1：异常不分层，永久性错误无故重试3次，造成资源浪费与卡顿**
     原始装饰器对所有异常统一重试，导致参数错误、文件不存在等永远不会恢复的错误依旧重试，造成资源浪费与卡顿。
     解决：做异常分层。区分「永久性错误（语法、参数、文件不存在）直接终止返回」和「临时错误（网络波动）延迟重试」，精准控制重试逻辑。
   - **问题2：参数化配置暴露隐藏边界 Bug**
     早期最大步数写死，模型一般2轮即可完成任务，边界问题从未暴露。新增 `--max-steps` 可配置参数后，出现「模型正常完成任务仍提示超限」的误判。
     解决：使用 `while...else` 语法区分循环退出场景，精准区分「模型主动结束」和「步数超限被动结束」，修复边界误判。
     工程感悟：可配置化会把原本不可能触发的边界问题，变成用户可随时触发的显性问题，倒逼代码健壮性提升。
   - **问题3：不规范提交导致公共仓库代码损坏**
     代码缩进错误未编译校验直接提交，导致仓库文件存在语法错误，持续3天未发现。
     解决：建立工程习惯，修改代码后强制执行 `py_compile` 编译校验，杜绝坏代码入版本库。

3. **项目总结与后续方向**
目前四大工具全部跑通，真实网络 API 验证重试机制有效性，装饰器调用链、闭包、wraps 原理全部吃透。
后续优化：替换 eval 实现、统一全局常量、精细化日志输出。

---

### 项目二：`basic_rag_pipeline.py` | 手写完整 RAG 链路
1. **实现内容**
不依赖任何 RAG 框架，纯手写内存版完整 RAG 链路：文本递归切分、重叠拼接、向量化、相似度检索、上下文组装，全程自主实现。

2. **核心难点：一次性修复4个静默 Bug**
所有 Bug 均为程序不报错、输出看似正常、底层逻辑失效的静默故障，极难排查。
- **递归切分静默降级**：`rfind` 匹配逻辑导致递归失效，实际降级为定长切分。当时并没有现成的检查工具，是靠人工按「每个块的结尾是否落在合法分隔符上」这个判据逐块核对才定位到的；这个人工判据后来才被固化成代码（即下方第 5 项指标）
- **尾部数据丢失**：提前返回逻辑绕过重叠合并，导致尾部文本截断丢失
- **生成空文本块**：缓冲区未做空判断直接刷新，产生无效空块
- **重叠上下文失效**：反向截断砍掉前置重叠内容，导致上下文拼接失效

3. **自研五维验收指标（核心亮点）**
解决「能跑但跑不对」的行业痛点，建立量化验收标准。**进度如实标注：当前第 1、2、3、5 项已落地为代码，第 4 项是已设计、待实现的计划。**

| # | 指标 | 状态 |
| --- | --- | --- |
| 1 | 重叠长度一致性校验 | ✅ **已实现**（`overlap=0` 场景未覆盖，见下） |
| 2 | 分块长度合规校验 | ✅ **已实现** |
| 3 | 原文内容零丢失校验 | ✅ **已实现**（`overlap=0` 场景未覆盖，见下） |
| 4 | 测试参数可追溯记录 | 📋 计划（待实现） |
| 5 | 分块边界必须落在合法分隔符上（唯一可检测递归降级的指标） | ✅ **已实现** |

第 5 项的落地形态是 `check_boundary(chunks, separators)`，返回「不合格块数 / 已检查块数 / 不合格下标」三元组，`__main__` 里用两道断言卡住：

- **哨兵断言** `checked_num > 0`：防止只分出 1 块时断言空跑。此处术语要分清——`checked_num=0` 时「不合格数 0」只是因为根本没检查过任何边界，属**假阴性（漏报，该报没报）**，而非假阳性（误报）；表现出来就是空跑导致的「假通过」
- **三元组整体断言** `boundary_result == (0, checked_num, [])`：所有非末尾块都必须以合法分隔符结尾
- 校验与分块共用同一份全局 `SEPARATORS`，避免两处分隔符定义漂移

**该指标已通过「故意注入 bug」的反向验证**——人为破坏分块逻辑后，断言确实报错拦下，证明它不是「永远通过」的摆设。它也是五项里唯一能检测递归切分静默降级的指标。

第 1 项的落地形态是 `check_overlap_consistency(chunks, overlap)`：逐对比较第 i 块末尾 `overlap` 个字符与第 i+1 块开头 `overlap` 个字符，返回「是否全部合法 / 错误对数 / 坏对下标」三元组（下标 i 表示 `chunks[i]` 与 `chunks[i+1]` 这一对校验失败），另配 `len(chunks) > 1` 哨兵断言防单块空跑。它与 `check_no_loss` 是互补关系：后者只看整体拼接能否还原，属宏观等价；前者能定位到具体哪一对相邻块的重叠区没对齐。

第 2 项的落地形态是 `check_max_chunk(chunks, max_chunk_size)`：取所有块中最大的字符长度与上限比对，返回「是否通过 / 最长块字数 / 最长块文本」三元组，`__main__` 中以断言卡住，超限时报出实际字数与上限。

第 1、2 项同样做了反向验证：人为改坏某块的结尾 → 边界校验与重叠一致性同时报警；注入一个 347 字的超长块 → 分块长度校验报警，而边界校验放行（它判的是结尾字符是否合法，与长度无关，两项判定口径不同、各管一摊）。

第 3 项的落地形态是 `check_no_loss(chunks, original, overlap)`：把每块的前缀重叠切掉后拼接还原全文，与原文逐字比对，返回「是否一致 / 还原长度 / 原文长度 / 首个差异说明」四元组；同样配了哨兵断言（`len(chunks) > 1`，防单块空跑）与一致性断言。第 4 个返回值是后补的，起因是一个盲区：最初只回两个长度，遇到「长度相同、内容被改」的情况会报出两个一模一样的数字（还原后长度 230 / 原文长度 230），断言确实拦下了，但报错信息里看不出问题在哪；现在会给出首个差异字符的下标及前后各 8 字的对照。

**已知缺口（如实记录）**：`overlap=0` 场景未覆盖。该值下重叠拼接（`prev[-overlap:]`）退化为取整个前一块、还原式（`chunk[overlap:]`）同步失效，实测还原出 263 字、原文 230 字，第 3 项断言会失败。第 1 项在该值下同样失真——`chunk[-0:]` 等价于取整块，比较对象从「重叠区」变成了「整个前一块 vs 空串」，于是报出「不匹配对 1」，属误报（数据并未真的违反重叠约定，是取切片的方式先失效了）。参数校验目前覆盖 `max_chunk_size<=0`、`overlap<0`、`overlap>=max_chunk_size` 三类非法输入，**`overlap==0` 是唯一没盖住的取值**。

4. **迭代优化思路**
第 1、2、3、5 项验收指标已落地（第 1、2、5 项含反向验证）。后续将递归切分改为迭代栈实现，解决超长文本递归深度溢出风险；补齐文档注释，固化测试用例；补上 `overlap=0` 的覆盖，并把第 4 项指标从计划补成代码。

---

### 项目三：`tool_agent_langgraph.py` | LangGraph 版 ReAct Agent
1. **实现内容**
在「项目一」手写 ReAct 的基础上，用 LangGraph `StateGraph` 重构调度循环：四个节点（`agent_node` 推理、`tool_node` 执行工具、`finish_done` 正常结束、`finish_over_limit` 超限结束）+ 一条条件边（`route_agent`），复用 `tool_agent.py` 中全部工具定义、工具映射、重试装饰器与最大步数配置，工具层零改动。另挂 `SqliteSaver` checkpointer 把状态落盘到脚本同目录的 `checkpoints.sqlite`，支持跨轮对话记忆。

2. **核心问题与解决思路**
   - **问题1：消息历史手动拼容易出错**
     项目一里每轮手动 `messages.append(...)` 维护完整历史；LangGraph 版声明 `Annotated[list, add_messages]`，节点只返回本轮增量消息，框架自动追加进历史，不用再手动拼列表。
   - **问题2：循环终止条件要显式落地成边**
     手写版靠 `while...else` 区分「模型主动结束 / 步数超限」；LangGraph 版把这两个终止条件收进 `route_agent` 条件边——无工具调用返回 `"done"`、达到 `DEFAULT_MAX_STEPS` 返回 `"over_limit"`、否则返回 `"tools"` 进入工具节点再回到推理节点，形成「推理-行动-观察」闭环。为让结束原因可追溯，两条终止路径分别接到专用结束节点 `finish_done` / `finish_over_limit`，各自写入 `finish_reason` 字段并在运行末尾打印，正常完成与超限结束一目了然。
   - **问题3：初始状态缺字段会抛 KeyError**
     `agent_node` 会读 `state["step"]`，运行末尾也会读 `state["finish_reason"]`，所以 `__main__` 里初始状态必须显式给 `step=0` 和 `finish_reason=""`，否则取值时报 KeyError。
   - **问题4：跨轮对话记忆需要持久化状态**
     没有 checkpointer 时，每次 `invoke` 都是独立会话，第二次调用不知道上一轮说过什么。挂上 checkpointer 后，用同一个 `thread_id` 即可让多次 `invoke` 共享状态，`messages` 经 `add_messages` 自动续上历史。`__main__` 里用第二次 invoke 问「我上一条问了你什么」，再断言 `len(state2["messages"]) > len(final_state["messages"])` 验证记忆确实生效（消息数增长，否则说明 checkpointer 没起效）。最初挂的是 `InMemorySaver`，但它的存档只活在进程内存里、脚本一结束记忆就没了，只适合调试；现改为 `SqliteSaver`，状态写进 SQLite 的 `checkpoints` / `writes` 两张表，同一个 `thread_id` 换个进程也能续上。
   - **问题5：存档路径用相对路径会「记忆凭空消失」**
     `sqlite3.connect("checkpoints.sqlite")` 是相对路径，而相对路径的基准是**运行时的工作目录（CWD）**、不是脚本所在目录：从 PyCharm 里运行、在终端 `cd` 到别处运行，会分别解析成不同的文件（`sqlite3.connect` 还会顺手新建一个空库），表现为「换了个地方启动，记忆就没了」。解决：`DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints.sqlite")`，用脚本自身位置锚定，从任何目录启动都读写同一份存档；会话标识也从硬编码提为顶部常量 `THREAD_ID`，为上服务时「从认证态派生」留出改造点。

3. **项目总结**
工具层零改动，只替换「调度循环」这一层：手写 `while` 循环 → 框架状态图；跨轮记忆等能力再交给框架自带的 checkpointer 白拿。两者对照能讲清 ReAct 的核心是「推理-行动-观察」闭环，具体用循环还是状态图实现是次要的。
