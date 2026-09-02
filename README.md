# ai-agent-learning
AI Agent、RAG学习练习项目的学习记录

## 项目介绍
手写练习AI Agent与RAG，底层逻辑全部手动实现，用于学习大模型应用开发。

## 项目模块
- simple_agent.py：简易对话客户端，实现对话记忆持久化（JSON）、历史截断、异常处理（早期在 AI 辅助下完成，正在重写为可独立讲解的版本）
- basic_rag_pipeline.py：内存版RAG完整链路，不依赖第三方向量库，分块/向量化/检索/组装全部手动实现（早期在 AI 辅助下完成，正在重写为可独立讲解的版本）
- tool_agent.py：无框架手写 ReAct Agent（DeepSeek API + Function Calling），循环执行工具调用直至给出答案，支持时间查询/数学计算/文件读取，带错误回传与 max_steps 防死循环

## RAG完整链路
原始文档 →文本分块 →Embedding向量化 →相似度检索 →拼接上下文 →Prompt组装 →调用LLM接口

## 已处理边界问题
1. 检索返回为空，上下文为空字符串兜底防护
2. top‑k参数越界保护
3. 上下文token预算：按模型窗口减去系统提示/用户问题/预留输出动态计算，防止超限

## 踩坑记录
（...）
## 依赖包
sentence‑transformers>=2.7.0
numpy>=1.26.0
tiktoken
requests
python-dotenv
