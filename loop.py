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

【边界（2026-11-12 更新）】
权限校验已移到 core/approval.py，轨迹落盘在本文件内完成；工具实现仍位于 core/tools.py。
会话状态只存在内存里，进程退出即丢；不含跨次运行的历史持久化。
Skill（技能）注入与确认卡都不改变这一条：它们没有引入任何可持久化的状态。

【接线后新增三步】
循环开始前：skills.system_message() 命中才注入一条 system 消息（只一次，不进 TOOLS）
每个写工具前：approval.propose() 出卡 → check_gate() 复核 → 拦下也要回填 tool 消息
每个写工具后：trace.write(type="receipt") 记落盘实测字节数与批准来源

【实测证据的三种新事件】
  action_card  确认卡被提出   card_id / name / fp / approved / approved_by
  blocked      门拦下未执行   card_id / name / why
  receipt      确认后实际发生 card_id / fp / status / before_bytes / after_bytes / md5
新增 type 不用改 core/trace.py —— 写入层不做白名单校验（trace.py:28）。
"""

import hashlib
import json
import os
import re
from dotenv import load_dotenv
from core import approval, skills, trace
from core.tools import TOOLS, run_tool

load_dotenv()

MAX_TURNS = 20                                # 强制退出上限，防止死循环烧钱


def _default_ask(messages, tools):
    """延迟 import（导入）模型层。

    🔴 顶层写 `from core.llm import ask` 时，core/llm.py 会在 import 阶段
    实例化 OpenAI client（客户端）；没配 key 的环境连「只列用例清单」都会炸。
    evals.py:28-31 的注释记过这个坑，这里从根上修掉：
    只有真的要调模型时才 import，桩驱动（stub）与离线评测完全不碰它。
    """
    from core.llm import ask
    return ask(messages, tools)


def _role(m):
    """messages（消息列表）里 dict（字典）和 ChatCompletionMessage 混着，
    取 role 要同时兼容两种：dict 用 .get()，对象用 getattr()。"""
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _target_path(name, arguments_json):
    """从工具参数里解析「这次写可能动到哪个文件」，供回执取实测字节数。

    只用于观测，不用于放行判断 —— 门看的是 fingerprint（指纹），不是路径。
    解析不出来就返回 None，回执里 after_bytes 会是 None，评测直接报 FAIL：
    拿不到实测值就不许算「执行成功」。
    """
    try:
        args = json.loads(arguments_json)
    except Exception:
        return None
    if name == "write_file":
        return args.get("path")
    if name == "run":
        # `copy /y a.txt b.txt` 这类命令：取最后一个像文件名的 token（词元）
        parts = str(args.get("command", "")).split()
        files = [p for p in parts if re.search(r"\.\w+$", p) and not p.startswith("/")]
        return files[-1] if files else None
    return None


def _bytes_of(path):
    """实测落盘字节数。文件不存在返回 None —— 不猜、不填 0。"""
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def run_agent(task: str, messages: list | None = None, trace_path=None,
              approver: str = "auto", ask_fn=None) -> str:
    """多轮驱动：模型自主决定工具调用次数，直到作答或达到轮数上限。

    approver  批准策略：eval（评测自动批准，标记 eval:auto）/ human（终端人工）
              / deny（永不批准，反向验证用）/ auto（无人值守，标记 auto:unattended）
    ask_fn    模型调用注入点。默认走 core.llm.ask；传桩函数即可离线跑完整主链，
              不联网、不需要 key，文件读写、轨迹、评测断言全是真的。
    """
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": task})
    trace_path = trace_path or trace.new_trace()          # 一次任务一个轨迹文件，与 messages 同生命周期
    ask_fn = ask_fn or _default_ask

    # ---------- Skill（技能）注入：只插一次，只进 messages，不进 TOOLS ----------
    skill_name, skill_msg = skills.system_message(task)
    if skill_msg and not any(_role(m) == "system" for m in messages):
        messages.insert(0, skill_msg)
        trace.write(trace_path, type="skill", turn=0, name=skill_name)

    for turn in range(1, MAX_TURNS + 1):
        # ---------- 本轮调用：获取模型意图 ----------
        msg = ask_fn(messages, TOOLS)
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
            name = tc.function.name
            arguments_json = tc.function.arguments
            risk = approval.risk_of(name)

            # ===== 写操作：先出确认卡，再过门 =====
            card = None
            if risk == "write":
                card = approval.propose(name, arguments_json, approver)
                trace.write(
                    trace_path, type="action_card", turn=turn,
                    card_id=card.card_id, name=name, fp=card.fp,
                    approved=card.approved, approved_by=card.approved_by,
                )
            ok, why = approval.check_gate(card, name, arguments_json)

            if not ok:
                # 🔴 门拦下仍必须回填 role="tool" 消息，tool_call_id 不能变。
                # 少了这一条，下一轮 invoke 会因为「assistant 的 tool_calls 没有
                # 对应的 tool 结果」直接报 400 —— 10-04 LangGraph 熔断那次踩过。
                result = f"错误：{why}（本次工具未执行）"
                trace.write(
                    trace_path, type="blocked", turn=turn,
                    name=name, card_id=card.card_id if card else None, why=why,
                )
                print(f"第 {turn} 轮：[门拦下] {name} —— {why}")
                # 这里不写 type="tool_call"：工具根本没执行，写了会让
                # tools_must 误判成「调用过」，制造假绿。
            else:
                target = _target_path(name, arguments_json) if risk == "write" else None
                before = _bytes_of(target)
                result = run_tool(name, arguments_json)
                after = _bytes_of(target)
                print(f"第 {turn} 轮：调用了 {name}")
                print("=== 工具真实返回 ===")
                print(result)
                print("工具输出指纹:", hashlib.md5(result.encode("utf-8")).hexdigest()[:12])
                # 写轨迹放在 print 之后、append 之前：这三件事都只依赖 result，
                # 顺序无关；但 trace 写失败不该阻断消息回填，所以它不是必经路径上的判断
                trace.write(
                    trace_path, type="tool_call", turn=turn,
                    name=name,
                    args=arguments_json,
                    md5=hashlib.md5(result.encode("utf-8")).hexdigest()[:12],
                    len=len(result),
                    result=trace.brief(result),
                    truncated=len(result) > trace.BRIEF_HEAD + trace.BRIEF_TAIL,
                )
                if risk == "write":
                    receipt = approval.make_receipt(card, result, before, after)
                    trace.write(trace_path, type="receipt", turn=turn, **receipt)
                    print(f"[回执] {receipt['card_id']} / {receipt['fp']} / "
                          f"落盘 {receipt['after_bytes']} 字节 / {receipt['approved_by']}")

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
        # REPL 有终端，走人工批准；评测/批处理走 approver="eval"，不会出现 input()
        run_agent(task, session, approver="human")
