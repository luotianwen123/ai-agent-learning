import requests
import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL ="https://api.deepseek.com/chat/completions"
MODEL="deepseek-chat"

def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def calculator(expression):
    try:
        result=eval(expression)
        return result
    except Exception as e:
        return f"error信息{e}"
def read_file(file_path):
    try:
        with open(file_path) as f:
            return f.read()
    except Exception as e:
        return f"错误信息是str{e}"

tools=[{"type":"function",
        "function":{
            "name":"get_current_time",
            "description":"获取当前系统时间，当用户询问现在几点、今天几号、当前日期时间时使用",
            "parameters":{
                "type":"object",
                "properties":{},
                "required": []
            }
        }
        },{"type": "function",
        "function": {
            "name": "calculator",
            "description": "用于数学计算，当用户需要进行算术运算、表达式求值时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "用于读取本地文件内容，当用户需要查看某个文件的内容时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件的绝对路径"
                    }
                },
                "required": ["file_path"]
            }},}
       ]

tool_map ={"get_current_time":get_current_time,"calculator":calculator,"read_file":read_file}

def call_llm(messages):
    try:
        body={"model":MODEL,
            "messages":messages,
            "tools":tools
              }
        resp=requests.post(
            OPENAI_BASE_URL ,
            json=body,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
        data=resp.json()
        reply=data["choices"][0]["message"]
        reply["done"]=not reply.get("tool_calls")
        return reply
    except Exception as e:
        print(f"调用大模型失败：{e}")
        raise
def run_agent(task):
    messages=[{"role":"user","content":task}]
    max_steps=5
    step=0
    while step<max_steps:
        step+=1
        print(f"\n--- 第 {step} 圈 ---")
        try:
            reply=call_llm(messages)
        except Exception as e:
            print(f"调用大模型失败：{e}")
            break

        messages.append(reply)
        if reply["done"]:
            print("回答完毕")
            break
        for tc in reply["tool_calls"]:
            name=tc["function"]["name"]
            args=json.loads(tc["function"]["arguments"])
            print(f"  执行工具：{name}，参数：{args}")
            result=tool_map[name](**args)
            messages.append({
                "role":"tool",
                "tool_call_id":tc["id"],
                "content":json.dumps(result,ensure_ascii=False),
            })
    if step>=max_steps:
        print("\n超限，强制结束")
    final = messages[-1].get("content", "")
    print(f"\n===== 最终答案 =====\n{final}")
    return final
if __name__ == "__main__":
    run_agent("现在几点")


