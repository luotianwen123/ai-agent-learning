# ai-agent-learning

三个项目，一条线：**从「会调 API」到「能自己写 Agent 内核」。**

| # | 项目 | 一句话 | 关键点 |
|---|---|---|---|
| **01** | [RAG 知识库问答服务](./01-rag-service/) | 把一个只读检索脚本做成**可评测、可服务化、有拒答**的问答服务 | 分块四道校验 · 双塔 vs 交叉重排对照实验 · 正负样本夹逼标定拒答阈值 · 4 指标评测线 + 存档对比 |
| **02** | [手写 ReAct Agent](./02-react-agent/) | 不用框架从零写 ReAct 循环，再用 LangGraph 重写一遍做对照 | Function Calling · 双出口终止 · 同工具连续 3 次熔断 · checkpointer 跨进程状态 |
| **03** | [mini-pi：极简 Coding Agent](./03-mini-pi/) | **自己写一个能改代码的 Agent 内核**，并给它配 trace 与回归集 | 多轮闭环 · 工具层（读/写/跑） · 轨迹落盘 · case-based 评测 |

---

## 为什么是这三个（能力递进，不是三个玩具）

```
01  RAG 服务       ── 先证明「我能把一个大模型功能做成工程件」：
                      有评测、有阈值、有服务化、有失败记录

02  ReAct Agent    ── 再证明「我懂 Agent 是怎么转起来的」：
                      循环、工具、终止、状态，全部手写一遍

03  mini-pi        ── 最后证明「我能造一个 Agent 内核」：
                      自己实现 loop + tools + trace + 评测，跑真实改代码任务
```

**三者共用一个方法论**：每次改动都回答三个问题 —— **改了什么 / 为什么 / 怎么验证**。

---

## 目录

```
01-rag-service/      RAG 知识库问答服务（Python + Chroma + FastAPI）
    README.md            项目详解：链路、评测线、19 张变更卡、踩坑
    RAG_main.py          一条命令入口（自带冷启动建库）
    FAST_API.py          HTTP 服务（/ask · /health）
    eval_rag.py          评测线（4 指标 + 存档 + 与上次对比）
    eval_runs/           评测存档（不可重建，必须入库 —— 见变更卡 18）

02-react-agent/      手写 ReAct Agent
    tool_agent.py        无框架版：Function Calling + 4 类工具
    tool_agent_langgraph.py  LangGraph 版：StateGraph + checkpointer

03-mini-pi/          Python 极简 Coding Agent + 轻量自进化
    loop.py              多轮闭环（两个终止条件）
    core/tools.py        工具层：读文件 / 写文件 / 跑命令
    core/trace.py        轨迹落盘
    core/evals.py        case-based 回归评测
    cases/               回归用例

practice/            学习期练习代码（分块边界、字典、闭包等），保留痕迹
```

---

## 最快跑通哪一个

**02（不需要 API Key 也能看结构，需要 Key 才能真跑）**
```bash
cd 02-react-agent
pip install -r ../01-rag-service/requirements.txt
python tool_agent.py "现在几点"
```

**01（自带示例语料，clone 下来不配也能建库）**
```bash
cd 01-rag-service
cp .env.example .env      # 填 OPENAI_API_KEY
python RAG_main.py "什么是 RAG"
```

**03**
```bash
cd 03-mini-pi
python loop.py
```

---

## 这个仓库最该看的三样东西

1. **`01-rag-service/README.md` 的变更卡**（19 张）
   每张都是「改了什么 / 为什么 / 怎么验证」。**包括改失败的**——
   比如两段式重排（rerank）实测 MRR 0.617 → 0.537，**负向，所以默认关闭并保留开关**。

2. **`01-rag-service/eval_runs/`**
   评测存档。**这些不是构建产物，不可重建**：同一配置换一份语料就是不同数字。
   所以它们必须入库——证据链不能断在最后一步（变更卡 18 就是这么来的）。

3. **`03-mini-pi/trace/`**
   每一次工具调用的轨迹。用来回答一个面试常问的问题：
   *「你说它跑通了，凭什么？」*
