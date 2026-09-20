from typing import TypedDict
class AgentState(TypedDict):
    messages:list[dict]
    step:int
    done:bool
def take_step(state: AgentState) -> AgentState:
    updated_step = state["step"] + 1
    new_message = {"role": "assistant","content": f"第{updated_step}轮响应"}
    updated_messages = state["messages"] + [new_message]
    return {"messages": updated_messages,"step": updated_step,"done": updated_step >= 3}
if __name__=="__main__":
    agent:AgentState={"messages": [{}],"step": 0,"done": False}
    agent=take_step(agent)
    print(f"step = {agent['step']}, done = {agent['done']}")
    print(len(agent["messages"]))
    agent = take_step(agent)
    print(f"step = {agent['step']}, done = {agent['done']}")
    print(len(agent["messages"]))
    agent = take_step(agent)
    print(f"step = {agent['step']}, done = {agent['done']}")
    print(len(agent["messages"]))