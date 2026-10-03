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


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int
    finish_reason: str


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


def tool_node(state: AgentState) -> dict:
    """
    工具执行节点：批量执行大模型发起的所有工具调用
    返回部分状态更新：所有工具执行结果消息
    """
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    tool_results = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]

        # 执行本地工具函数，完全复用 tool_map 与重试机制
        result = tool_map[tool_name](**tool_args)

        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result)
        })

    return {"messages": tool_results}


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
    return {"finish_reason": "正常结束：模型不再请求工具，推理已完成"}


def finish_over_limit(state: AgentState) -> dict:
    print(f"===== 超限结束：推理步数达到上限 {DEFAULT_MAX_STEPS} =====")
    return {"finish_reason": f"超限结束：步数达到上限 {DEFAULT_MAX_STEPS}，被强制终止"}


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("finish_done", finish_done)
graph.add_node("finish_over_limit", finish_over_limit)


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
graph.add_edge("tools", "agent")
graph.add_edge("finish_done", END)
graph.add_edge("finish_over_limit", END)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # 初始状态必须显式给 step=0，否则 agent_node 取值会报 KeyError
    initial_state = {
        "messages": [{"role": "user", "content": "现在几点？"}],
        "step": 0,
        "finish_reason":""
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
         "finish_reason": ""},
        config
    )
    print("第 1 次消息条数:", len(final_state["messages"]))
    print("第 2 次消息条数:", len(state2["messages"]))
    print("\n=== 第 2 次 invoke 的模型回答 ===")
    print(state2["messages"][-1].content)
    assert len(state2["messages"]) > len(final_state["messages"]), "记忆没生效：两次消息数一样"