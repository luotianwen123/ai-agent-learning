"""
tool_agent_langgraph.py
基于LangGraph StateGraph实现的完整ReAct工具调用Agent
复用 tool_agent.py 中全部工具定义、工具映射、重试装饰器与最大步数配置
使用 LangChain ChatOpenAI 客户端对接 DeepSeek 大模型，通过状态图实现自动推理循环
状态用 SqliteSaver 持久化，thread_id 为会话标识
支持能力：时间查询、数学计算、本地文件读取、城市天气查询
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import sqlite3
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from tool_agent import tools, tool_map, DEFAULT_MAX_STEPS

# 存档库路径：锚定到本文件所在目录，避免"相对路径跟着运行时的工作目录跑"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints.sqlite")
# 会话标识：以后上服务时改成从认证态派生（用户ID + 会话ID），不要接受前端随便传
THREAD_ID = "demo-2"

MAX_REPEAT=3


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int
    finish_reason: str
    last_tool: str
    repeat_count: int
    repeat_breach: bool


def call_llm(messages):
    """调用绑定了工具的大模型，返回 AIMessage 对象"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    llm_with_tools = llm.bind_tools(tools)
    reply = llm_with_tools.invoke(messages)
    return reply


def agent_node(state: AgentState) -> dict:
    """
    Agent 推理节点：调用大模型生成回复，步数+1
    返回部分状态更新：新的助手消息 + 更新后的步数
    """
    reply = call_llm(state["messages"])
    return {
        "messages": [reply],
        "step": state["step"] + 1
    }


def tool_sig(name: str, args: dict) -> str:
    """
    把（工具名 + 参数）压成一个签名，用来判断"是不是同一个调用"。
    👉 只想按"工具名"算连续，就把函数体改成 return name（一行切换）
    """
    return name + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True)


def tool_node(state: AgentState) -> dict:
    """
    工具执行节点：批量执行大模型发起的所有工具调用
    新增：同一签名连续超过 MAX_REPEAT 次时熔断（第 MAX_REPEAT+1 次不执行）
    """
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    # 取成本地变量，在循环里自己往前推（一轮可能有 N 个调用，不能每轮重读 state）
    last_tool = state.get("last_tool", "")
    repeat_count = state.get("repeat_count", 0)

    tool_results = []
    breach = False

    for i, tc in enumerate(tool_calls):
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]

        # 判断这次算不算"连续同一个调用"
        sig = tool_sig(tool_name, tool_args)
        if sig == last_tool:
            nxt = repeat_count + 1          # 连续同一个 → +1
        else:
            last_tool, nxt = sig, 1         # 换了一个 → 从 1 重新数

        # 执行前判限：第 MAX_REPEAT+1 次就熔断，不执行
        if nxt > MAX_REPEAT:
            breach = True
            repeat_count = nxt              # 记成"第 4 次请求"（请求了但没执行）
            # 关键：把本轮剩下的（含这次）都补一条"未执行"的工具消息。
            # 否则 assistant.tool_calls 与 tool 消息数量不匹配，下次调模型会被 API 判 400
            for rest in tool_calls[i:]:
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": rest["id"],
                    "content": f"未执行：触发重复调用熔断（{rest['name']} 连续调用超过上限 {MAX_REPEAT} 次）",
                })
            break

        # 通过 → 落地计数并执行
        repeat_count = nxt
        result = tool_map[tool_name](**tool_args)
        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        })

    return {
        "messages": tool_results,
        "last_tool": last_tool,
        "repeat_count": repeat_count,
        "repeat_breach": breach,
    }


def route_after_tools(state: AgentState) -> str:
    """tools 节点出口：熔断了直接去收尾，否则回 agent 继续推理"""
    return "breach" if state.get("repeat_breach") else "continue"


def finish_repeat_limit(state: AgentState) -> dict:
    """第三个收尾节点：重复调用熔断（与前两个对称）"""
    sig = state.get("last_tool", "")
    n = state.get("repeat_count", 0)
    print("===== 熔断结束：同一工具连续调用超限 =====")
    print(f"调用签名: {sig}")
    print(f"已连续调用 {n} 次，上限 {MAX_REPEAT} 次 → 第 {n} 次请求未执行")
    return {
        "finish_reason": f"熔断结束：{sig} 连续调用 {n} 次，超过上限 {MAX_REPEAT}",
        "repeat_count": 0,        # 重置（时机②）
        "last_tool": "",
        "repeat_breach": False,   # 清标记，否则下次 invoke 会误判
    }


def route_agent(state: AgentState) -> str:
    """
    Agent 节点出口路由函数：判断下一步走向
    返回值对应 path_map 中的三个键：
    - "done":       模型不再发起工具调用，推理正常完成
    - "over_limit": 步数达到上限，强制终止
    - "tools":      继续进入工具节点执行调用
    """
    last_msg = state["messages"][-1]
    # 终止条件1：模型没有发起工具调用，推理已完成
    if not last_msg.tool_calls:
        return "done"
    # 终止条件2：已达到最大推理步数，强制终止
    if state["step"] >= DEFAULT_MAX_STEPS:
        return "over_limit"
    # 继续循环：执行工具后回到下一轮推理
    return "tools"


def finish_done(state: AgentState) -> dict:
    print("=====正常结束：模型已给出最终回答=====")
    return {
        "finish_reason": "正常结束：模型不再请求工具，推理已完成",
        "repeat_count": 0, "last_tool": "", "repeat_breach": False,
    }


def finish_over_limit(state: AgentState) -> dict:
    print(f"===== 超限结束：推理步数达到上限 {DEFAULT_MAX_STEPS} =====")
    return {
        "finish_reason": f"超限结束：步数达到上限 {DEFAULT_MAX_STEPS}，被强制终止",
        "repeat_count": 0, "last_tool": "", "repeat_breach": False,
    }


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("finish_done", finish_done)
graph.add_node("finish_over_limit", finish_over_limit)
graph.add_node("finish_repeat_limit", finish_repeat_limit)


graph.add_edge(START, "agent")


graph.add_conditional_edges(
    source="agent",
    path=route_agent,
    path_map={
        "done": "finish_done",
        "over_limit": "finish_over_limit",
        "tools": "tools"
    }
)
graph.add_conditional_edges(
    source="tools",
    path=route_after_tools,
    path_map={
        "breach": "finish_repeat_limit",
        "continue": "agent",
    }
)
graph.add_edge("finish_repeat_limit", END)
graph.add_edge("finish_done", END)
graph.add_edge("finish_over_limit", END)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # 初始状态必须显式给 step=0，否则 agent_node 取值会报 KeyError
    initial_state = {
        "messages": [{"role": "user", "content": "现在郑州天气？"}],
        "step": 0,
        "finish_reason": "",
        "last_tool": "",
        "repeat_count": 0,
        "repeat_breach": False,
    }

    # 一键 invoke 跑完整循环
    config = {"configurable": {"thread_id": THREAD_ID}}
    final_state = app.invoke(initial_state, config)

    print("=== 第 1 次 invoke 结束后的状态 ===")
    print(f"总推理步数: {final_state['step']}")
    print(f"结束原因: {final_state['finish_reason']}")
    print("\n=== 第 1 次 invoke 的模型回答 ===")
    print(final_state["messages"][-1].content)

    print("\n=== 第 2 次 invoke：同一个 thread_id ===")
    state2 = app.invoke(
        {"messages": [{"role": "user", "content": "我上一条问了你什么？"}],
         "step": 0,
         "finish_reason": "",
         "last_tool": "",
         "repeat_count": 0,
         "repeat_breach": False,
         },
        config
    )
    print("第 1 次消息条数:", len(final_state["messages"]))
    print("第 2 次消息条数:", len(state2["messages"]))
    print("\n=== 第 2 次 invoke 的模型回答 ===")
    print(state2["messages"][-1].content)
    assert len(state2["messages"]) > len(final_state["messages"]), "记忆没生效：两次消息数一样"