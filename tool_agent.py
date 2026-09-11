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

# 工具名(字符串)→函数映射：模型只返回工具名，靠这张表翻译成真正可调用的函数
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
        resp.raise_for_status()  # 非 2xx 状态码立刻抛异常，避免拿错误的响应继续执行
        data=resp.json()  # 反序列化：JSON 文本→Python 字典/列表，之后才能按键取值
        reply=data["choices"][0]["message"]  # 嵌套取值：choices 是列表(可能有多个候选)，取第一个的 message
        reply["done"]=not reply.get("tool_calls")  # 自定义终止开关：没有工具调用则结束循环；tool_calls 是可选键(想调工具时才存在)，用 .get() 避免 KeyError
        return reply
    except Exception as e:  # 工具层只记录不处理，具体应对交给上层
        print(f"调用大模型失败：{e}")
        raise
def run_agent(task):
    # 消息历史是"列表套字典"：列表保存按顺序排列的每条消息；字典里 role 记谁说的、content 记内容
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

        # 追加而非覆盖：每轮回复接到历史尾部，下一轮模型才能读到完整上下文
        messages.append(reply)
        if reply["done"]:
            print("回答完毕")
            break
        # tool_calls 是列表：模型一轮可能同时请求多个工具调用，逐个遍历执行
        for tc in reply["tool_calls"]:
            name=tc["function"]["name"]
            # arguments 是 JSON 字符串，json.loads 转成 Python 字典后才能取参、传参
            args=json.loads(tc["function"]["arguments"])
            print(f"  执行工具：{name}，参数：{args}")
            # **args 拆包为关键字参数；此行即 ReAct 的 A(行动)：执行工具，返回值即观察结果
            result=tool_map[name](**args)
            messages.append({
                "role":"tool",
                # 回填模型下发的 tool_call_id：API 靠它把结果和对应调用对上
                "tool_call_id":tc["id"],
                # json.dumps 序列化(与 resp.json() 相反)：content 只收字符串，故把结果转成 JSON；ensure_ascii=False 让中文不转义
                "content":json.dumps(result,ensure_ascii=False),
            })
    if step>=max_steps:
        print("\n超限，强制结束")
    # 负索引取最后一条消息；.get("content","") 缺省返回空串，防止最后没内容时崩溃
    final = messages[-1].get("content", "")
    print(f"\n===== 最终答案 =====\n{final}")
    return final
if __name__ == "__main__":
    run_agent("现在几点")
