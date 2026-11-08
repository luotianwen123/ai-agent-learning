"""loop.py —— 多轮 Agent 闭环

把「一次往返」升级为「while 式多轮循环」：模型自己决定调几次工具、调哪些、
什么时候停。核心只有一个新概念：终止条件，而且有两个——

1. 正常退出：模型这一轮没返回 tool_calls，说明它认为信息够了，直接作答
2. 强制退出：轮数到达 MAX_TURNS（默认 20），防止死循环烧钱

【执行顺序】
1. 把本次任务作为 user 消息追加到 messages（可由调用方传入既有会话）
2. 进入 for turn in range(1, MAX_TURNS + 1) 循环
3. 每轮先 ask()，把 msg 无条件塞进 messages（先塞再判断）
4. 若 msg 无 tool_calls：打印轮数并返回 msg.content（正常出口）
5. 否则逐个执行 tool_call，结果以 role="tool" 回填，打印调用日志
6. 循环跑满仍未 return：打印达到最大轮数（强制出口）

【数据流】
进：__main__ 的 REPL 从标准输入读到的任务文本
中转：messages 列表，只增不减，是全部上下文的唯一载体
出：stdout
副作用：读写磁盘（经由工具）

【工具输出指纹】
md5(工具返回值)[:12]。模型回答措辞每次不同，不可作回归判据；
工具返回值由本地代码产生，具有确定性，指纹不变才说明重构没有改变行为。
基线：0a8c1b0d5dfb

【边界】
不含权限校验、轨迹落盘。工具实现位于 core/tools.py。
会话状态只存在内存里，进程退出即丢；不含跨次运行的历史持久化。
"""

import hashlib
from dotenv import load_dotenv
from core import trace
from core.llm import ask
from core.tools import TOOLS, run_tool

load_dotenv()

MAX_TURNS = 20                                # 强制退出上限，防止死循环烧钱


def run_agent(task: str, messages: list | None = None, trace_path=None) -> str:
    """多轮驱动：模型自主决定工具调用次数，直到作答或达到轮数上限。

    messages 由调用方传入时，本次任务追加到既有会话上 —— 模型能看见前几轮
    问过什么、调过哪些工具。不传则本次自建，函数返回即丢弃（一次性任务）。
    """
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": task})
    trace_path = trace_path or trace.new_trace()          # 一次任务一个轨迹文件，与 messages 同生命周期

    for turn in range(1, MAX_TURNS + 1):
        # ---------- 本轮调用：获取模型意图 ----------
        msg = ask(messages, TOOLS)
        messages.append(msg)                  # 无条件先塞再判断，后续 tool 消息要对应它的 tool_call_id

        # ---------- 正常出口：没有 tool_calls，模型认为信息够了 ----------
        if not msg.tool_calls:
            print(f"第 {turn} 轮：模型作答")
            print("=== 模型最终回答 ===")
            print(msg.content)
            trace.write(trace_path, type="final", turn=turn, chars=len(msg.content or ""))
            return msg.content

        # ---------- 执行本轮所有 tool_call，结果回填消息列表 ----------
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            print(f"第 {turn} 轮：调用了 {tc.function.name}")
            print("=== 工具真实返回 ===")
            print(result)
            print("工具输出指纹:", hashlib.md5(result.encode("utf-8")).hexdigest()[:12])
            # 写轨迹放在 print 之后、append 之前：这三件事都只依赖 result，
            # 顺序无关；但 trace 写失败不该阻断消息回填，所以它不是必经路径上的判断
            trace.write(
                trace_path, type="tool_call", turn=turn,
                name=tc.function.name,
                args=tc.function.arguments,
                md5=hashlib.md5(result.encode("utf-8")).hexdigest()[:12],
                len=len(result),
            )
            messages.append({
                "role": "tool",               # 固定为 tool，写 user 会被 API 拒绝
                "tool_call_id": tc.id,        # 必须与上一条 tool_calls 中的 id 完全一致，否则 400
                "content": result,            # 必须是 str；传 dict 会在序列化阶段失败
            })

    # ---------- 强制出口：循环跑满仍未 return ----------
    print(f"达到最大轮数 {MAX_TURNS}，已停止")
    trace.write(trace_path, type="max_turns", turn=MAX_TURNS)
    return f"达到最大轮数 {MAX_TURNS}，已停止"


if __name__ == "__main__":
    # 外层 REPL：run_agent 在「模型不调工具」时返回，但那不等于任务完成 ——
    # 模型也可能是信息不足在反问。循环本身分不清这两种，只有人能判断。
    # 所以把「答完继续」这件事交给外层：内层跑完一次任务并交回回答，外层决定下一步。
    #
    # session 必须活在整个 REPL 里：它一销毁，模型就看不见上一轮说了什么，
    # 你回答它的问题它也读不懂。
    session = []
    print("输入任务后回车；空行、exit 或 退出 结束。")
    while True:
        try:
            task = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task or task.lower() in ("exit", "quit") or task == "退出":
            break
        run_agent(task, session)
