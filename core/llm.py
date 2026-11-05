"""core/llm.py —— 全项目唯一和模型说话的地方

【角色】
内核层最底下一块，模型的「嘴和耳朵」。往上给 loop.py 提供 ask()，往下对接 DeepSeek 的 HTTP 接口。
任何模块想调模型都必须经过这里，不许自己 new 一个 client。

【工作流】
1. import 时读 .env，建一个全局 _client（只建一次，不是每次调用都建）
2. 调用方给 messages + tools
3. 发请求给 DeepSeek，拿回一条回复
4. 剥出 choices[0].message 交回去

【数据流】
进：messages —— 字典列表，每条形如 {"role": "user", "content": "..."}
    tools    —— 工具说明书列表（JSON Schema）；不传 = 纯聊天，模型不会调工具
出：message  —— ChatCompletionMessage 对象，两种可能：
    · 有 .tool_calls  → 模型想调工具（参数在 .function.arguments 里，是字符串不是字典）
    · 只有 .content   → 模型直接回答了
调用方：loop.py 的每一轮

【已知欠账】
resp 上还有 .usage（token 计数，含 prompt_cache_hit_tokens —— DeepSeek 靠它算钱），
但当前只 return 了 message，usage 被丢掉了。第 4 天写循环要统计成本时，改这里即可。
「改一处就够」正是把调用收口到这个文件的意义。

【不负责】
不重试、不超时、不记日志、不做降级。这些第 2 周加，但都要加在这个文件里。
"""

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
