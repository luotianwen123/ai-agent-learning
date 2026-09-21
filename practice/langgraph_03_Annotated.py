"""	传 init_state	                              传 {}
① 读出记忆	step=3, messages=3 条, done=True	  同左
② 叠加你的输入	step=0 → 替换成 0	              不覆盖
messages=[] → 合并，仍 3 条	                      不合并
done=False → 替换成 False	                      不覆盖 → True
③ 节点跑几次	done=False → 跑 3 次	                  done=True → 跑 1 次
④ 最终	step=3 · messages=6	                      step=4 · messages=10"""
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph,END,START
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
#done = 循环的续行/终止判断(False/Ture)
#START = 入口
#route_func = 真正做决定的人
class AgentState(TypedDict):
    messages: Annotated[list,add_messages]
    step:int
    done:bool
def step_node(state:AgentState)->dict:
    new_step=state['step']+1
    new_msg={"role":"assistant","content":f"执行第{new_step}步"}
    new_done=(new_step>=3)
    return {
        "step":new_step,
        "messages":[new_msg],
        "done":new_done}
def route_func(state:AgentState)->str:
    if state["done"]:
        return END
    else:
        return "step"
g=StateGraph(AgentState)
g.add_node("step",step_node)
g.add_edge(START,"step")
g.add_conditional_edges(source="step",path=route_func,path_map={END:END,"step":"step"})
if __name__ == "__main__":
    init_state = {
        "messages": [],
        "step": 0,
        "done": False
    }
    with SqliteSaver.from_conn_string("checkpoints.db")as cp:
        app=g.compile(checkpointer=cp)
        out = app.invoke(init_state, config={"configurable": {"thread_id": "user_1"}})
        print(f"step: {out['step']}")
        print(f"done: {out['done']}")
        print(f"len(messages): {len(out['messages'])}")
