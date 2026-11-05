"""loop.py —— Agent 的最小闭环（单轮）

【角色】
项目当前唯一能跑的驱动脚本。它把「一次问答」变成「一次干活」：
模型说要调工具 → 我的代码真的去执行 → 结果喂回去 → 模型给出最终回答。
位于内核层中间：上面是入口 内核.py，下面调 core/llm.py 和 core/tools.py。

⚠️ 现状：只有 1 个来回，不是多轮循环。第 4 天升级成 core/loop.py 的 while 多轮后，本文件并入。

【工作流】
1. 建 messages 列表，放一条 user 消息
2. 第 1 次问模型（ask）→ 拿到 tool_calls
3. 按 tool_call 里的名字和参数，真的执行本地函数
4. 把执行结果作为 role="tool" 的消息塞回 messages（tool_call_id 必须和上面对上）
5. 第 2 次问模型 → 模型拿着真实内容作答
6. 打印结果

【数据流】
进：硬编码的一句提问（"请读取 a.txt ..."）
核心结构：messages 列表 —— 只增不减，是所有上下文的唯一载体
出：控制台打印
副作用：读了磁盘上的 a.txt

【为什么要有「工具输出指纹」】
hashlib.md5(工具返回值)[:12]。模型每次回答的措辞都不一样，不能当回归判据；
工具返回值是你的代码产生的，是确定性的 —— 指纹不变才证明重构没改坏。
当前基线：0a8c1b0d5dfb

【不负责】
不做多轮循环（第 4 天）、不管权限（第 2 周）、不落盘 transcript（第 3 周）。
工具函数本身写在 core/tools.py，本文件只负责调用。
"""

import hashlib
import os
import json
from dotenv import load_dotenv
from core.llm import ask

load_dotenv()


TOOLS = [                                       # 和 check_llm.py 完全一样，照抄
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录里的一个文本文件，返回内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"],
            },
        },
    }
]


def read_file(path):
    """真的去打开文件，读完返回字符串"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_tool(name, arguments_json):
    """按工具名分发。arguments_json 是字符串，先用 json.loads 转成字典"""
    args = json.loads(arguments_json)          # 字符串 → 字典
    if name == "read_file":
        return read_file(args["path"])         # args["path"] 就是 a.txt
    return "错误：没有这个工具 " + name        # 兜底：模型点名了不存在的工具


messages = [
    {"role": "user", "content": "请读取 a.txt 这个文件，然后告诉我里面写了什么。"}
]

# ---------- 第 1 次调用：模型点单 ----------

msg = ask(messages, TOOLS)

if not msg.tool_calls:
    print("模型没调工具，直接答了：", msg.content)
    raise SystemExit(0)                        # 提前结束，后面不用跑了

messages.append(msg)                           # 把模型这条原样存回去（它带着 tool_calls）

# ---------- 我的代码执行工具，把结果塞回消息列表 ----------
for tc in msg.tool_calls:
    result = run_tool(tc.function.name, tc.function.arguments)
    print("=== 工具真实返回 ===")
    print(result)
    print("工具输出指纹:", hashlib.md5(result.encode("utf-8")).hexdigest()[:12])
    messages.append({
        "role": "tool",                        # 角色是 tool，不是 user
        "tool_call_id": tc.id,                 # 必须等于上面那个 tc.id，错一个字符就 400
        "content": result,                     # 必须是字符串，不能塞 dict
    })

# ---------- 第 2 次调用：模型拿着真实内容说话 ----------
resp2_msg = ask(messages, TOOLS)

print("=== 模型最终回答 ===")
print(resp2_msg.content)
