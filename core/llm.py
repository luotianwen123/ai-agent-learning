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

【2026-11-09 更新：超时与重试已加】
原本这里写「不含超时」，理由是「能力后续统一加」。代价在当天就来了：
A-6 对照实验跑到一半静默挂了 11 分钟，现场看起来像死机 ——
根因是 OpenAI SDK 的默认 timeout 是 600 秒，而代码里一个参数都没传。
现在显式设 timeout=90、max_retries=1。
⇒ 教训：「先不加，以后再统一加」这句话，在评测链路上等于给调试埋雷。
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "deepseek-chat"          # deepseek-reasoner 不支持 function calling，不可替换

# 采样温度：本项目固定 0，理由不是「低温更好」，而是「评测要能比」。
# temperature 缺省值是 1.0（服务端的默认，不是 0）。不写这一个参数，
# 就把「同一用例跑三次结果不同」的责任从模型推到了评测层 —— 而评测层
# 恰恰是靠「三次结果是否一致」来判断稳定性的，两边一起抖就分不清谁的责任。
#
# 提成模块常量的原因：写死在 ask() 的调用里，就没法做对照实验
# （同一批用例分别用 0 和 1 跑，才知道波动到底来自采样还是模型本身）。
TEMPERATURE = 0

_client = OpenAI(                # 下划线前缀：模块内部使用，外部不要直接引用
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    # 🔴 必须显式设超时。OpenAI SDK 的默认值是 **600 秒**，不是 30 秒也不是 60 秒。
    # 2026-11-09 实测代价：A-6 的对照实验跑到第 5 次任务，某一轮请求挂住，
    # 整个脚本静默等了 11 分钟没动静 —— 现场看起来像"卡死"，实际是默认超时在倒计时。
    # 评测失败要快、要响，不能一挂十分钟让人以为程序死了。
    timeout=90.0,
    # 默认会重试 2 次。保留 1 次应付网络抖动，但不能再多：
    # 重试次数多了，一次真故障会被包装成「慢」，评测报告里就看不出发生过错误。
    max_retries=1,
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
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message
