"""core/llm.py —— 模型调用层，全项目唯一的模型出入口

【职责】
向上为 loop 提供 ask()，向下对接 DeepSeek 的兼容接口。
其他模块不得自行实例化 client；换模型、加超时、记用量都只改此文件。

【执行顺序】
1. 模块导入时读取 .env 并构建全局 _client（只构建一次，不随每次调用重建）
2. 接收 messages 与 tools
3. 发起请求，取回首条回复
4. 剥离 choices[0].message 返回

【接口】
进：messages —— 字典列表，元素形如 {"role": "user", "content": "..."}
    tools    —— 工具 Schema 列表；不传则为纯对话，模型不会发起工具调用
出：message  —— ChatCompletionMessage，两种形态：
    · 含 .tool_calls  → 模型请求调用工具，参数在 .function.arguments 中，为 JSON 字符串
    · 仅含 .content   → 模型直接作答
调用方：loop 的每一轮

【已知限制】
响应对象 resp 上还有 .usage（token 用量，含 prompt_cache_hit_tokens，DeepSeek 按此计费），
当前仅返回 message，usage 未被保留。需要统计成本的场景应在此处扩展，
这正是把调用集中到单一出入口的意义。

【边界】
不含重试、超时、日志、降级。这些能力后续统一加在本文件中，不分散到调用方。
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "deepseek-chat"          # deepseek-reasoner 不支持 function calling，不可替换

_client = OpenAI(                # 下划线前缀：模块内部使用，外部不要直接引用
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def ask(messages, tools=None):
    """向模型发起一次请求，返回首条回复。

    messages: 消息列表（字典的列表）
    tools:    工具 Schema 列表；不传则为纯对话，模型不会发起工具调用
    """
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=0,
    )
    return resp.choices[0].message
