# ai-agent-learning
AI Agent、RAG学习练习项目，手写Demo

## 项目介绍
手写练习AI Agent与RAG，底层逻辑全部手动实现，用于学习大模型应用开发。

## 项目模块
- simple_agent.py：简易对话客户端，实现对话记忆持久化（JSON）、历史截断、异常处理
- basic_rag_pipeline.py：内存版RAG完整链路，不依赖第三方向量库，分块/向量化/检索/组装全部手动实现
- tool_agent.py：无框架手写 ReAct Agent（DeepSeek API + Function Calling），循环执行工具调用直至给出答案，支持时间查询/数学计算/文件读取，带错误回传与 max_steps 防死循环

## RAG完整链路
原始文档 →文本分块 →Embedding向量化 →相似度检索 →拼接上下文 →Prompt组装 →调用LLM接口

## 已处理边界问题
1. 检索返回为空，上下文为空字符串兜底防护
2. top‑k参数越界保护
3. 上下文token预算：按模型窗口减去系统提示/用户问题/预留输出动态计算，防止超限

## 踩坑记录

### 坑 1：分块 overlap 无限膨胀
**现象**：当时看到的 chunk 越拼越大，随着递归深度的增加，同一段文本会被重复复制多次，最终输出的块总数远超预期。
**原因**：因为recursive_split在递归过程中反复对同一段文本叠加overlap，导致块边界内容被无限复制膨胀。
**解决**：分块与重叠逻辑一体化实现，拼接后超限时截断（2026-08-28 提交修复）。

### 坑 2：LLM 返回的 JSON 解析失败未处理
**现象**：当时是因为api版本变更，接口返回错误结构（限流，内容过滤，服务异常），返回内容不是合法json触发的
**原因**：requests 拿到的响应不保证是合法 JSON，直接 resp.json() 会抛 JSONDecodeError。
**解决**：llm_chat 中将 resp.json() 包进 try/except，解析失败抛出带上下文的 RuntimeError。

### 坑 3：HuggingFace 下载模型网络超时
**现象**：sentence-transformers 初始化模型时连接 huggingface.co 失败，报 ConnectionError / SSL timeout。
**原因**：国内网络直连 HuggingFace 主站不稳定，模型文件下载被阻断。
**解决**：设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`，让 sentence-transformers 走国内镜像站下载模型权重。

### 坑 4：PyCharm 全局变量警告
**现象**：IDE 在函数内部引用外部定义的变量（如 `API_KEY`）时标黄提示 "Access to a protected member" 或 "Shadows name from outer scope"。
**原因**：函数内未声明 `global` 却读写外层变量，IDE 静态分析认为存在作用域混淆风险。
**解决**：将配置变量改为通过函数参数传入，或用 `os.getenv` + `.env` 加载，消除全局状态依赖。

### 坑 5：空列表 join 导致上下文为空
**现象**：检索返回空列表时，`"".join(chunks)` 得到空字符串，拼进 prompt 后 LLM 看不到任何上下文。
**原因**：`str.join` 对空列表的语义就是空字符串，本身不是 bug，但下游没有做空结果兜底。
**解决**：join 后判断结果是否为空，若为空则塞入兜底提示（如"未检索到相关内容"），或在检索层做空结果保护。

### 坑 6：eval 表达式求值的安全隐患
**现象**：calculator 工具使用 `eval()` 直接执行用户输入的算式字符串，代码注入风险高。
**原因**：`eval()` 可执行任意 Python 代码，恶意输入能操控文件系统、发起网络请求等。
**解决**：本地学习场景保留 eval 以便快速验证功能；生产环境必须替换为 ast.literal_eval 或手写表达式解析器。

## 依赖包
sentence‑transformers>=2.7.0
numpy>=1.26.0
tiktoken
requests
python-dotenv
