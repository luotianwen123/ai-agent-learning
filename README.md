# ai-agent-learning
AI Agent、RAG学习练习项目，手写Demo

## 项目介绍
手写练习AI Agent与RAG，底层逻辑全部手动实现，用于学习大模型应用开发。

## 项目模块
- simple_agent.py：简易对话客户端，实现对话记忆持久化（JSON）、历史截断、异常处理
- basic_rag_pipeline.py：内存版RAG完整链路，不依赖第三方向量库，分块/向量化/检索/组装全部手动实现

## RAG完整链路
原始文档 →文本分块 →Embedding向量化 →相似度检索 →拼接上下文 →Prompt组装 →调用LLM接口

## 已处理边界问题
1. 检索返回为空，上下文为空字符串兜底防护
2. top‑k参数越界保护
3. 上下文token预算：按模型窗口减去系统提示/用户问题/预留输出动态计算，防止超限

## 踩坑记录

### 坑 1：分块 overlap 无限膨胀
**现象**：【你来填：当时看到的 chunk 越拼越大，具体表现是？】
**原因**：【你来填：提示——每块前面拼接上一块尾部时，长度是怎么累加的？】
**解决**：分块与重叠逻辑一体化实现，拼接后超限时截断（2026-08-28 提交修复）。

### 坑 2：LLM 返回的 JSON 解析失败未处理
**现象**：【你来填：当时是什么触发的？】
**原因**：requests 拿到的响应不保证是合法 JSON，直接 resp.json() 会抛 JSONDecodeError。
**解决**：llm_chat 中将 resp.json() 包进 try/except，解析失败抛出带上下文的 RuntimeError。

### 待填（遇到就记）
- HuggingFace 下载模型网络问题（镜像源）
- 全局变量警告
- join 空列表得到空字符串

## 依赖包
sentence‑transformers>=2.7.0
numpy>=1.26.0
tiktoken
requests
