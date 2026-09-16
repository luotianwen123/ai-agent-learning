from argparse import ArgumentParser
import requests
import datetime
import time
import json
import os
from functools import wraps
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL ="https://api.deepseek.com/chat/completions"
MODEL="deepseek-chat"

# 当前工具都是本地操作，retry 主要面向未来的外部 API 工具
def retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for i in range(3):
            try:
                return func(*args, **kwargs)
            except (TypeError,json.JSONDecodeError,FileNotFoundError,NameError,SyntaxError) as e:
                return f"【{func.__name__}】永久性错误，不重试：{e}"
            except Exception as e:
                last_error = e
                print(f"第{i+1}次失败:{e}")
                time.sleep(i+1)
        return f"【{func.__name__}】重试 3 次仍失败：{last_error}"
    return wrapper

@retry
def get_current_time()->str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@retry
def calculator(expression:str)->int|float|str:
    return eval(expression)

@retry
def read_file(file_path:str)->str:
    with open(file_path) as f:
        return f.read()

@retry
def get_weather(city:str)->str:
    resp=requests.get(f"https://wttr.in/{city}?format=j1",timeout=5)
    resp.raise_for_status()
    data:dict=resp.json()
    temp=data["current_condition"][0]["temp_C"]
    weatherdesc=data["current_condition"][0]["weatherDesc"][0]["value"]
    return f"{city}:{temp}°C,{weatherdesc}"

tools=[
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"获取当前系统时间，当用户询问现在几点、今天几号、当前日期时间时使用",
            "parameters":{
                "type":"object",
                "properties":{},
                "required": []
            }
        }
    },
    {
        "type": "function",
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
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description":"查询指定城市的当前温度与天气，当用户询问某地天气时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":{
                        "type": "string",
                        "description":"要查询天气的城市名"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 工具名(字符串)→函数映射：模型只返回工具名，靠这张表翻译成真正可调用的函数
tool_map ={
    "get_current_time":get_current_time,
    "calculator":calculator,
    "read_file":read_file,
    "get_weather":get_weather
}

def call_llm(messages):
    try:
        body={
            "model":MODEL,
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

DEFAULT_MAX_STEPS = 5

def run_agent(task,max_steps=DEFAULT_MAX_STEPS):
    messages=[{"role":"user","content":task}]
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
    else:
        print("\n超限，强制结束")

    final = messages[-1].get("content", "")
    print(f"\n===== 最终答案 =====\n{final}")
    return final

if __name__ == "__main__":
    parser=ArgumentParser(description="ReAct 工具调用Agent，支持时间查询、计算器、读本地文件、天气查询")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="用户输入的问题"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS, 
        help="Agent最大推理轮次"
    )
    args = parser.parse_args()
    if args.question is None:
        parser.print_help()
    else:
        run_agent(args.question,max_steps=args.max_steps)
