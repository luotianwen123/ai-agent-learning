#messages 是追加；
#但这不是默认行为，是因为加了 Annotated[list, add_messages] 才变成追加。
#如果去掉这个注解，LangGraph 会默认直接覆盖。
#tool_agent.py 里不用是因为我手动在每一轮更新对话后加入messages信息
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int
    done: bool
def step_node(state: AgentState) -> dict:
    new_step = state["step"] + 1
    new_msg = {"role": "assistant", "content": f"执行第 {new_step} 步"}
    new_done = (new_step >= 3)
    return {
        "step": new_step,
        "messages": [new_msg],
        "done": new_done
    }
def route_func(state: AgentState)->str:
    if state["done"] :
        return END
    else:
        return "step"
g = StateGraph(AgentState)
g.add_node("step", step_node)
g.add_edge(START, "step")
g.add_conditional_edges(
    source="step",
    path=route_func,
    path_map={END: END, "step": "step"}
)
app = g.compile()
if __name__ == "__main__":
    init_state = {
        "messages": [],
        "step": 0,
        "done": False
    }
    out = app.invoke(init_state)
    print(f"step: {out['step']}")
    print(f"done: {out['done']}")
    print(f"len(messages): {len(out['messages'])}")
