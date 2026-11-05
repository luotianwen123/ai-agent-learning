# core/llm.py
# 全项目唯一调用模型的地方。换模型、加超时、加日志，只改这个文件。

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "deepseek-chat"          # 换模型只改这里

_client = OpenAI(                # 前面的下划线 = 模块内部用，外部别直接碰它
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def ask(messages, tools=None):
    """把消息列表发给模型，返回第一条回复。

    messages: 消息列表（字典组成的列表）
    tools:    工具说明书列表；不传就是纯聊天，模型不会调工具
    """
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    return resp.choices[0].message
