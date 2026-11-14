"""core/approval.py —— 确认卡 / 权限门 / 回执（视频方向第 ② 步落地）

【它最终解决什么】
模型说「我已经写好了」不能当证据。确认卡管的是「人同意了什么」，
回执管的是「同意之后实际发生了什么」。两者分开，评测才不会自证。

【执行顺序】
propose() 出卡 → approver（批准策略）签字 → check_gate() 在使用前复核 →
run_tool() 执行 → make_receipt() 记下实测值 → trace.write(type="receipt")

【三个不可省的机制】
1. fingerprint()（指纹）：对「工具名 + 参数」算 sha256（哈希）取前 12 位。
   批准后参数被改，指纹对不上，卡立刻失效 —— 这叫 TOCTOU
   （Time-Of-Check To Time-Of-Use，检查与使用之间被插队）。
   只比工具名不比参数，等于没有门。
   ⚠️ tc.function.arguments（工具参数）是 **JSON 字符串**，不是 dict（字典）：
   必须先 json.loads（解析）再 sort_keys（按键排序）重新序列化，
   否则模型把参数顺序换一下，指纹就变，会被误判成「卡失效」。
2. approved_by（批准来源）：评测时用自动批准，必须写成 "eval:auto"，
   不能写成 "human" —— 轨迹是证据，把自动批准记成人批准就是伪造证据。
3. deny（拒绝）策略：永远不批准，供反向验证用。
   判据：走 deny 时，写工具一次都不许执行，磁盘一个字节都不许变。

【边界】
本模块只判断，不执行工具、不写文件、不写轨迹 —— 副作用全留在 loop.py。
只读工具（read_file）不需要卡：它不产生不可逆副作用。
未知工具默认按「写」处理：fail-closed（拿不准就拦），不是 fail-open（拿不准就放）。
"""

import hashlib
import json
import uuid
from dataclasses import dataclass


# 工具名 → 风险级别。loop.py 据此决定要不要出确认卡。
# run 归入写：它执行 shell（命令解释器）命令，可以 copy / del / 改文件。
RISK = {
    "read_file": "read",
    "write_file": "write",
    "run": "write",
}


def risk_of(name):
    """查工具风险。未知工具返回 "write"（写）—— 拿不准就拦，不许放行。"""
    return RISK.get(name, "write")


def fingerprint(name, arguments_json):
    """工具名 + 参数 → 12 位指纹。参数顺序无关，内容一字相关。"""
    try:
        args = json.loads(arguments_json)
    except Exception:
        args = {"_raw": arguments_json}          # 非法 JSON 也要有稳定指纹，不能抛
    raw = json.dumps(
        {"name": name, "args": args},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


@dataclass(frozen=True)
class ActionCard:
    """确认卡。frozen=True（冻结）＝ 建完不许改字段，
    想「批准」只能造一张新卡，避免有人原地改 approved 把证据抹掉。"""
    card_id: str
    name: str
    arguments: str
    fp: str
    approved: bool = False
    approved_by: str = "none"


def propose(name, arguments_json, approver="auto"):
    """生成确认卡并按策略签字。

    approver 四种取值：
      eval   评测用，自动批准，标记 "eval:auto"（明确不是人）
      human  终端人工，调 input()；超时/EOF 一律按拒绝
      deny   永不批准，反向验证用
      auto   无人值守默认，标记 "auto:unattended"

    🔴 只有 human 分支允许 input()。评测是 3 条 × 3 轮无人值守批量跑，
    任何一条路径上出现 input() 都会把整批挂死。
    """
    card_id = "card-" + uuid.uuid4().hex[:8]
    fp = fingerprint(name, arguments_json)

    approved = False
    approved_by = "none"
    if approver == "deny":
        approved_by = "deny:reverse-test"
    elif approver == "eval":
        approved, approved_by = True, "eval:auto"
    elif approver == "human":
        print(f"\n[确认卡 {card_id}] 工具 {name} 参数 {arguments_json[:200]}")
        try:
            ans = input("批准执行？输入 y 回车 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        approved = ans in ("y", "yes")
        approved_by = "human" if approved else "human:denied"
    else:
        approved, approved_by = True, "auto:unattended"

    return ActionCard(
        card_id=card_id,
        name=name,
        arguments=arguments_json,
        fp=fp,
        approved=approved,
        approved_by=approved_by,
    )


def check_gate(card, name, arguments_json):
    """权限门：在使用工具**之前**复核。返回 (是否放行, 原因文本)。

    只读工具直接放行；写工具要卡已签字、工具名没换、指纹没变。
    第三条就是 TOCTOU 防护：批准之后到执行之间参数被插队改掉，卡必须失效。
    """
    if risk_of(name) == "read":
        return True, "只读工具自动放行"
    if card is None or not card.approved:
        return False, "写操作未经批准（确认卡没签字）"
    if card.name != name:
        return False, "批准后工具名被改，卡已失效"
    if card.fp != fingerprint(name, arguments_json):
        return False, "批准后参数被改，卡已失效（TOCTOU）"
    return True, "门已通过"


def make_receipt(card, result, before_bytes=None, after_bytes=None):
    """回执：确认之后**实际**发生了什么。

    after_bytes 必须是 os.path.getsize()（实测落盘字节数），不是模型自述、
    不是工具返回值里的字符串。拿不到就是 None，评测直接判 FAIL ——
    没有实测值不许算「执行成功」。
    md5 / len 记的是工具返回值，用于两次运行对比是否逐字节一致。
    """
    return {
        "card_id": card.card_id,
        "fp": card.fp,
        "name": card.name,
        "status": "executed",
        "approved_by": card.approved_by,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "md5": hashlib.md5(result.encode("utf-8")).hexdigest()[:12],
        "len": len(result),
    }
