# ai-agent-learning
AI Agent、RAG学习练习项目，手写Demo
# ai‑agent‑learning
## 项目介绍
手写练习AI Agent与RAG，底层逻辑全部手动实现，用于学习大模型应用开发。

## 项目模块
- simple_agent：简易智能体，实现对话记忆持久化、token控制、异常处理
- rag_memory_demo：内存版RAG，不依赖第三方向量库

## RAG完整链路
原始文档 →文本分块 →Embedding向量化 →相似度检索 →拼接上下文 →Prompt组装 →调用LLM接口

## 已处理边界问题
1. 检索返回为空，上下文为空字符串兜底防护
2. top‑k参数越界保护

## 踩坑记录
（这里留给你后续慢慢填：huggingface网络镜像、全局变量警告、join空列表得到空字符串等）

## 依赖包
sentence‑transformers
numpy
tiktoken
requests
