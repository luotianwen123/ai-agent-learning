"""check_llm.py —— 存档：验证 DeepSeek 会不会返回 tool_calls

【角色】
不属于主链路。它是项目第一天的一块地基证据 ——
证明在动手写任何循环之前，先确认了模型能吐出结构化的工具调用。
面试时能说：「我第一步先验证了 tool_calls 的原始格式，而不是先堆框架。」

⚠️ 禁止改造。它保留的是原始形态。一旦改成走 core/llm.py 的 ask()，
   「原始格式长什么样」这个证据就没了。只读存档和活代码要分开。

【工作流】
单次调用，不循环。打印 content、tool_calls，以及 arguments 的类型。

【数据流】
进：一条 user 消息 + 一个 read_file 工具说明书
出：控制台打印（content / tool_calls / name / arguments / 类型）

【它验证出的三条事实（2026-11-05 实测）】
1. tool_calls 不为 None，function.name = read_file
2. arguments 是 JSON 字符串，类型是 <class 'str'>，不是字典 ——
   所以后面所有地方都必须 json.loads()
3. content 是空的（模型没闲聊，直接调工具），这是正常行为，不是 bug

【不负责】
不做任何真实文件操作。它只证明「模型会点单」，不证明「菜真的端上来了」。
后者是 loop.py 的活。
"""


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
