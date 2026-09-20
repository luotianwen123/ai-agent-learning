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
- `practice/`：日常练习归档目录，按「专题_序号_名称」命名，不再散落在 PyCharm 工程里（当前：LangGraph 三练）

### 📦 项目依赖
- `sentence-transformers>=2.7.0`
- `numpy>=1.26.0`
- `tiktoken`
- `requests`
- `python-dotenv`

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

pip install "sentence-transformers>=2.7.0" "numpy>=1.26.0" tiktoken requests python-dotenv
```

### 3. 配置 API Key
三个模块均调用 DeepSeek API（模型 `deepseek-chat`）。在项目根目录新建 `.env` 文件：

```dotenv
OPENAI_API_KEY=你的_DeepSeek_API_Key
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

⚠️ 两点注意：
- 首次运行会自动下载 embedding 模型 `BAAI/bge-small-zh-v1.5`（约 100MB），需要联网等待。
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

### 5. 常见问题
| 现象 | 原因与处理 |
| --- | --- |
| `OPENAI_API_KEY 未配置，请在 .env 文件中填写` | `.env` 缺失或变量名为 `OPENAI_API_KEY` 拼写有误 |
| `配置文件不存在: config.json` | `simple_agent_demo.py` 未创建 `config.json`，见上方步骤 ③ |
| 首次运行卡住 / 下载缓慢 | 正在拉取 embedding 模型，确认网络可访问 HuggingFace |
| 调用报网络错误 | 脚本对临时性网络错误做了延迟重试；持续失败请检查 API Key 余额与网络代理 |

---

## ✨ 项目深度复盘（简历 / 面试 完整版）
> 更新时间：2026-09-16 | 素材来源：真实 Git 提交记录 + 本地版本比对，无虚构、可核验

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
- **递归切分静默降级**：`rfind` 匹配逻辑导致递归失效，实际降级为定长切分，依靠「边界校验指标」才定位问题
- **尾部数据丢失**：提前返回逻辑绕过重叠合并，导致尾部文本截断丢失
- **生成空文本块**：缓冲区未做空判断直接刷新，产生无效空块
- **重叠上下文失效**：反向截断砍掉前置重叠内容，导致上下文拼接失效

3. **自研五维验收指标（核心亮点）**
解决「能跑但跑不对」的行业痛点，建立量化验收标准：
   1. 重叠长度一致性校验
   2. 分块长度合规校验
   3. 原文内容零丢失校验
   4. 测试参数可追溯记录
   5. 分块边界必须落在合法分隔符上（唯一可检测递归降级的指标）

4. **迭代优化思路**
当前全部指标自测通过。后续将递归切分改为迭代栈实现，解决超长文本递归深度溢出风险；补齐文档注释，固化测试用例。
