"""check_llm.py —— 初始验证存档：确认模型能返回结构化 tool_calls

【状态】
DEPRECATED。保留为项目起点的验证记录，不参与主链路，请勿改造。
一旦改写为调用 core/llm.py 的 ask()，其记录的「原始返回结构」即失效。

【用途】
在搭建任何循环之前，先确认模型具备工具调用能力。
只做单次请求，打印 content、tool_calls 及 arguments 的类型。

【数据流】
进：一条 user 消息 + 一个 read_file 工具 Schema
出：stdout（content / tool_calls / function.name / arguments / 类型）

【初始验证结论（2026-11-05 实测）】
1. tool_calls 非 None，function.name 为 read_file
2. arguments 是 JSON 字符串，类型为 str 而非 dict —— 后续所有消费方都必须 json.loads()
3. content 为空（模型未作答而是直接发起工具调用），属正常行为

【边界】
不做真实文件操作。此处只验证「模型是否发起调用」，不验证「调用是否被执行」。
后者由 loop.py 负责。
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录里的一个文本文件，返回内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读的文件路径，相对于工作目录",
                    }
                },
                "required": ["path"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model="deepseek-chat",         # deepseek-reasoner 不支持 function calling
    messages=[
        {"role": "user", "content": "请读取 a.txt 这个文件，然后告诉我里面写了什么。"}
    ],
    tools=TOOLS,
)

msg = resp.choices[0].message

print("=== content ===")
print(msg.content)                 # 发起工具调用时为空，属正常
print("=== tool_calls ===")
print(msg.tool_calls)

if msg.tool_calls:
    for tc in msg.tool_calls:
        print("id        =", tc.id)
        print("name      =", tc.function.name)
        print("arguments =", tc.function.arguments, type(tc.function.arguments))
        print("parsed    =", json.loads(tc.function.arguments))
