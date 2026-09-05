# ai-agent-learning

个人学习记录。从 Python 基础补起，逐步手写实现 AI Agent 与 RAG 的核心链路。

> 说明：仓库中的早期代码有一部分是在 AI 辅助下完成的，我正在逐个重写成能独立讲解的版本。下面的模块描述只写目前能讲清楚的部分，讲不清的等重写完再补回来。

## 项目模块

- `simple_agent_demo.py`：简易对话客户端，实现对话记忆持久化（JSON）、历史截断、异常处理。（早期在 AI 辅助下完成，正在重写）
- `basic_rag_pipeline.py`：内存版 RAG 链路练习，包含分块 / 向量化 / 检索 / 组装。（早期在 AI 辅助下完成，正在重写）
- `tool_agent.py`：无框架手写 ReAct Agent（DeepSeek API + Function Calling），实现工具调用循环。

##代码使用目的 

basic_rag_pipeline.py中_raw参数的使用--递归切分的时候，如果每一层都做 overlap 拼接，递归越深，重叠的部分就被叠加越多次，最后一块文本里大部分都是重复内容。所以我用 _raw 标记'我是递归进来的，你别再叠了'——overlap 只在最外层拼一次。顺带参数校验也只做一次，不用每层重复。


## 学习日志

`learning-log.md`：每天的练习内容、卡点和次日计划，持续更新。

## 依赖包

sentence-transformers>=2.7.0
numpy>=1.26.0
tiktoken
requests
python-dotenv
