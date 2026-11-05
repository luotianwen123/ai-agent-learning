"""loop.py —— 多轮 Agent 闭环

把「一次往返」升级为「while 式多轮循环」：模型自己决定调几次工具、调哪些、
什么时候停。核心只有一个新概念：终止条件，而且有两个——

1. 正常退出：模型这一轮没返回 tool_calls，说明它认为信息够了，直接作答
2. 强制退出：轮数到达 MAX_TURNS（默认 20），防止死循环烧钱

【执行顺序】
1. 构造 messages，放一条 user 消息
2. 进入 for turn in range(1, MAX_TURNS + 1) 循环
3. 每轮先 ask()，把 msg 无条件塞进 messages（先塞再判断）
4. 若 msg 无 tool_calls：打印轮数并返回 msg.content（正常出口）
5. 否则逐个执行 tool_call，结果以 role="tool" 回填，打印调用日志
6. 循环跑满仍未 return：打印达到最大轮数（强制出口）

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
不含权限校验、轨迹落盘。工具实现位于 core/tools.py。
"""

import hashlib
from dotenv import load_dotenv
from core.llm import ask
from core.tools import TOOLS, run_tool

load_dotenv()

MAX_TURNS = 20                                # 强制退出上限，防止死循环烧钱


def run_agent(task: str) -> str:
    """多轮驱动：模型自主决定工具调用次数，直到作答或达到轮数上限。"""
    messages = [
        {"role": "user", "content": task}
    ]

    for turn in range(1, MAX_TURNS + 1):
        # ---------- 本轮调用：获取模型意图 ----------
        msg = ask(messages, TOOLS)
        messages.append(msg)                  # 无条件先塞再判断，后续 tool 消息要对应它的 tool_call_id

        # ---------- 正常出口：没有 tool_calls，模型认为信息够了 ----------
        if not msg.tool_calls:
            print(f"第 {turn} 轮：模型作答")
            print("=== 模型最终回答 ===")
            print(msg.content)
            return msg.content

        # ---------- 执行本轮所有 tool_call，结果回填消息列表 ----------
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            print(f"第 {turn} 轮：调用了 {tc.function.name}")
            print("=== 工具真实返回 ===")
            print(result)
            print("工具输出指纹:", hashlib.md5(result.encode("utf-8")).hexdigest()[:12])
            messages.append({
                "role": "tool",               # 固定为 tool，写 user 会被 API 拒绝
                "tool_call_id": tc.id,        # 必须与上一条 tool_calls 中的 id 完全一致，否则 400
                "content": result,            # 必须是 str；传 dict 会在序列化阶段失败
            })

    # ---------- 强制出口：循环跑满仍未 return ----------
    print(f"达到最大轮数 {MAX_TURNS}，已停止")
    return f"达到最大轮数 {MAX_TURNS}，已停止"


if __name__ == "__main__":
    run_agent("请读取 a.txt 这个文件，然后告诉我里面写了什么。")
