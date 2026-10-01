"""
tool_agent_lg.py
基于LangGraph StateGraph实现的完整ReAct工具调用Agent
复用 tool_agent.py 中全部工具定义、工具映射、重试装饰器与最大步数配置
使用 LangChain ChatOpenAI 客户端对接 DeepSeek 大模型，通过状态图实现自动推理循环
支持能力：时间查询、数学计算、本地文件读取、城市天气查询
"""

from dotenv import load_dotenv
load_dotenv()

import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from tool_agent import tools, tool_map, DEFAULT_MAX_STEPS


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int


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
    返回值对应 path_map 中的键：
    - "end": 终止运行（推理完成或步数超限）
    - "tools": 进入工具节点执行调用
    """
    last_msg = state["messages"][-1]
    # 终止条件1：模型没有发起工具调用，推理已完成
    if not last_msg.tool_calls:
        return "end"
    # 终止条件2：已达到最大推理步数，强制终止
    if state["step"] >= DEFAULT_MAX_STEPS:
        return "end"
    # 继续循环：执行工具后回到下一轮推理
    return "tools"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    source="agent",
    path=route_agent,
    path_map={
        "end": END,
        "tools": "tools"
    }
)
graph.add_edge("tools", "agent")
app = graph.compile()


if __name__ == "__main__":
    # 初始状态必须显式给 step=0，否则 agent_node 取值会报 KeyError
    initial_state = {
        "messages": [{"role": "user", "content": "现在几点？"}],
        "step": 0
    }

    # 一键 invoke 执行完整循环，替代原来的手动逐节点调用
    final_state = app.invoke(initial_state)

    print("=== 最终状态统计 ===")
    print(f"总消息条数: {len(final_state['messages'])}")
    print(f"总推理步数: {final_state['step']}")
    print("\n=== 模型最终回答 ===")
    print(final_state["messages"][-1].content)
