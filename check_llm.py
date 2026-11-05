# check_llm.py
# 目的：验证 DeepSeek 会不会返回「结构化的工具调用」（tool_calls）
# 这是整个 Agent 循环唯一的数据通路，不通就别往下写
# 【存档】2026-11-05 第一次验证 tool_calls 格式的历史脚本，不参与主链路，禁止改造


import os                          # 用来读环境变量
import json                        # 用来把 JSON 字符串转成字典
from dotenv import load_dotenv     # 把 .env 里的键值对读进环境变量
from openai import OpenAI          # DeepSeek 兼容 OpenAI 格式，所以用它

load_dotenv()                      # 不带参数：从当前文件所在目录往上逐级找 .env

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),     # 从环境变量取 key
    base_url=os.getenv("OPENAI_BASE_URL"),   # https://api.deepseek.com
)

TOOLS = [                          # 列表：可以给模型多个工具，先给 1 个
    {
        "type": "function",        # 工具类型是「函数」
        "function": {
            "name": "read_file",     # 工具名：模型靠这个名字点单
            "description": "读取工作目录里的一个文本文件，返回内容",   # 什么时候该用它
            "parameters": {                                   # 参数说明书（JSON Schema）
                "type": "object",
                "properties": {                               # 有哪些参数
                    "path": {
                        "type": "string",
                        "description": "要读的文件路径，相对于工作目录",
                    }
                },
                "required": ["path"],                         # 必填的参数名
            },
        },
    }
]

resp = client.chat.completions.create(
    model="deepseek-chat",         # 不能用 deepseek-reasoner，它不支持工具调用
    messages=[
        {"role": "user", "content": "请读取 a.txt 这个文件，然后告诉我里面写了什么。"}
    ],
    tools=TOOLS,                   # 把说明书交给模型
)

msg = resp.choices[0].message      # 取第一条回复

print("=== content ===")
print(msg.content)                 # 模型的文字回答，可能为 None
print("=== tool_calls ===")
print(msg.tool_calls)              # 模型想调哪些工具，不调就是 None

if msg.tool_calls:                 # 非空 = 模型确实要点单
    for tc in msg.tool_calls:      # tc = tool call（一次工具调用）
        print("id        =", tc.id)
        print("name      =", tc.function.name)
        print("arguments =", tc.function.arguments, type(tc.function.arguments))
        print("parsed    =", json.loads(tc.function.arguments))
