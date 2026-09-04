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
        resp.raise_for_status()#将要出现的HTTP 状态码、异常问题立刻抛出，不这么写我们的代码会拿着错误的响应继续执行，短期内脚本可以跑，但是出错时会加大我们的排查成本
        data=resp.json()#反序列化、JSON 字符串将其变成python的字典/列表，不写它，你拿到的只是文本，没法用 `data["键"]` 取值；写了它，才能像操作普通字典一样操作接口数据。
        reply=data["choices"][0]["message"] #一层层打开嵌套结构、挖到最深处的值（列表索引、嵌套取值），choices 是列表：因为能返回多个答案，有序集合
                                            #data["choices"]["message"] → TypeError（列表不能用字符串索引）
                                            #data["message"] → KeyError（顶层没有这个键）
        reply["done"]=not reply.get("tool_calls")#首先左边用reply["done"]是用 [] 赋值，新增一个我们自己定义的键 done，done 的作用标记"这一轮还要不要继续循环"——是整个 ReAct 循环的终止开关;不用.get() ，是因为.get()不能当赋值目标，右边是我们读可选字段使用的，当读取的字段键不存在是我们用[]硬取会抛 KeyError崩溃，而.get没传第二个参数会返回none not 在这里不是取反，是整个 ReAct 循环的终止开关,是用来判断有没有工具调用的场景出现读取可选字段：模型想调工具时才有这个键，不给工具时根本不存在。用 [] 硬取 → KeyError；.get() 没传第二参数 → 返回 None
        return reply
    except Exception as e:#89-91我们工具层只负责记录，不负责处理（业务层根据具体要求处理），不写的话我们只会知道系统报错，但具体是什么错误，哪里出错我们不知情
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


