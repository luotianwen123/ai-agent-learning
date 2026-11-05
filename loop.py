"""loop.py —— 单轮 Agent 闭环

把「一次问答」变成「一次任务」：模型给出工具调用意图，本地代码执行，
结果回传后再由模型作答。当前只有一次往返；多轮驱动见 core/loop.py。

【执行顺序】
1. 构造 messages，放一条 user 消息
2. 第一次 ask()：模型返回 tool_calls
3. 按 name + arguments 执行本地工具
4. 结果以 role="tool" 回填 messages，tool_call_id 与上一步一一对应
5. 第二次 ask()：模型基于真实结果作答

【数据流】
进：硬编码的提问
中转：messages 列表，只增不减，是全部上下文的唯一载体
出：stdout
副作用：读取磁盘

【工具输出指纹】
md5(工具返回值)[:12]。模型回答措辞每次不同，不可作回归判据；
工具返回值由本地代码产生，具有确定性，指纹不变才说明重构没有改变行为。
基线：0a8c1b0d5dfb

【边界】
不含多轮循环、权限校验、轨迹落盘。工具实现位于 core/tools.py。
"""

import hashlib
from dotenv import load_dotenv
from core.llm import ask
from core.tools import TOOLS, run_tool

load_dotenv()


messages = [
    {"role": "user", "content": "请读取 a.txt 这个文件，然后告诉我里面写了什么。"}
]

# ---------- 第一次调用：获取工具调用意图 ----------

msg = ask(messages, TOOLS)

if not msg.tool_calls:
    print("模型未调用工具，直接回答：", msg.content)
    raise SystemExit(0)

messages.append(msg)                           # 必须回填：后续 tool 消息要对应它的 tool_call_id

# ---------- 执行工具，结果回填消息列表 ----------
for tc in msg.tool_calls:
    result = run_tool(tc.function.name, tc.function.arguments)
    print("=== 工具真实返回 ===")
    print(result)
    print("工具输出指纹:", hashlib.md5(result.encode("utf-8")).hexdigest()[:12])
    messages.append({
        "role": "tool",                        # 固定为 tool，写 user 会被 API 拒绝
        "tool_call_id": tc.id,                 # 必须与上一条 tool_calls 中的 id 完全一致，否则 400
        "content": result,                     # 必须是 str；传 dict 会在序列化阶段失败
    })

# ---------- 第二次调用：模型基于真实结果作答 ----------
resp2_msg = ask(messages, TOOLS)

print("=== 模型最终回答 ===")
print(resp2_msg.content)
