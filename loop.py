# loop.py
# 目的：跑通「模型说要调工具 → 我的代码真去执行 → 结果喂回模型 → 模型拿真实内容回答」
# 这是 Agent 的最小闭环：只有一个来回，还不是多轮
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
