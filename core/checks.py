"""core/checks.py —— 评测中台的断言注册表

【它最终解决什么】
以前加一类断言要改 core/evals.py 的 check() 那个大函数，而 fail-open（失效即放行）、
fail-closed（缺证据判失败）、tools_must 超集语义全挤在里面 —— 改的人越多，
越容易把别人的防护顺手踩掉。现在断言是**可注册的小函数**：
新增一类 = 写一个函数 + 一个装饰器，主链一行不动。
这就是「可不断迭代更新、适配度高」的机制本体，不是一句口号。

【契约】
每个 checker（断言器）签名固定：
    (row, case, expect, events) -> [(断言名, bool, 说明), ...]
  row     一次运行的事实：sandbox / answer / trace / checks
  case    用例定义（含 _dir，指向用例目录）
  expect  case["expect"]；checker 只读自己那一个字段
  events  已读好的轨迹事件列表（统一读一次，不重复开文件）
@register(名字, 触发字段)：只有该字段在 expect 里为真才跑；
触发字段为 None 表示每次都调用，由函数自己决定要不要产出断言。

【四条不许丢的防护】
1. 轨迹缺失 ⇒ 单独报 FAIL（fail-closed）；不能让 tools_forbidden 判「一次都没调 = 通过」
2. tools_must 用超集语义（按序包含、允许额外调用），不是精确相等
3. 没有回执 ≠ 没有违规；回执五条要核内容，不只判「有没有」
4. 断言函数不许抛异常：run_all 逐个兜住，异常转成一条 FAIL，不让整批变 ERROR

【边界】
只读。不改磁盘、不调模型、不写轨迹。统一入口与报告在 core/evals.py。
"""

import json
from pathlib import Path


REGISTRY = []


def register(name, field=None):
    """注册一个 checker。field 是 case["expect"] 里触发它的字段名。"""
    def deco(fn):
        REGISTRY.append({"name": name, "field": field, "fn": fn})
        return fn
    return deco


def list_checks():
    """当前注册了哪些断言器 —— 报告里带上它，改没改断言集一眼可查。"""
    return [{"name": i["name"], "field": i["field"]} for i in REGISTRY]


def read_events(trace_path):
    """读 JSONL（每行一个事件）轨迹。读不到、坏行一律跳过 —— 不抛异常。

    断言函数不许抛：轨迹正被占用、进程被 kill 时文件可能写到一半，
    但那不该把整条用例变成 ERROR —— 该报 FAIL 就报 FAIL。
    """
    if not trace_path:
        return []
    events = []
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # 半行（写一半被 kill）直接跳过
    except OSError:
        return []
    return events


def _check_order(need, called):
    """need 是否**按顺序**出现在 called 里，允许中间插别的调用。

    superset（超集）语义：该做的一个不漏、先后不乱，但长度不必相等。
    精确相等会把「多调一次自查」这种好行为判成失败，测试立刻变 brittle（易碎）。
    """
    pos = 0
    for name in need:
        try:
            pos = called.index(name, pos) + 1
        except ValueError:
            # 失败必须说清是「没调」还是「调了但顺序不对」，两者修法完全不同
            if name in called:
                return False, f"顺序不对：{name} 出现在它该出现的位置之前 —— 实际 {called}"
            return False, f"缺少 {name} —— 实际只调用了 {called}"
    return True, f"必须按序包含 {need}，实际 {called}"


@register("files", "files")
def check_files(row, case, expect, events):
    """环境终态：磁盘上真有、且逐字节相同，才算数。模型自述不算。"""
    results = []
    sandbox = Path(row["sandbox"]) if row.get("sandbox") else None
    if sandbox is None:
        return results
    for item in expect.get("files", []):
        path = item.get("path", "")
        # 拼「沙箱绝对路径 + 相对文件名」，与 cwd 无关：
        # 项目根目录下恰好也有一份 a.txt / b.txt，用相对路径会查错地方
        target = sandbox / path
        want = bool(item.get("exists", True))
        actual_exists = target.exists()
        results.append((
            f"files:{path}:exists",
            actual_exists == want,
            f"期望{'存在' if want else '不存在'}，实际{'存在' if actual_exists else '不存在'}",
        ))
        if not actual_exists or not want:
            continue
        ref_name = item.get("same_bytes_as")
        if not ref_name:
            continue
        ref = (Path(case["_dir"]) / "fixture" / ref_name).read_bytes()
        actual = target.read_bytes()
        results.append((
            f"files:{path}:bytes",
            actual == ref,
            f"期望与 {ref_name} 逐字节相同，实际 {len(actual)} 字节 / 参考 {len(ref)} 字节",
        ))
    return results


@register("tools", None)
def check_tools(row, case, expect, events):
    """路径不变量：该调的一个不漏、禁调的一次不许出现、怎么停的也要查。"""
    tools_must = expect.get("tools_must") or []
    tools_forbidden = expect.get("tools_forbidden") or []
    stop = expect.get("stop")
    if not (tools_must or tools_forbidden or stop):
        return []

    results = []
    called = [e.get("name") for e in events if e.get("type") == "tool_call"]

    # 🔴 轨迹缺失必须单独报 FAIL。否则 called = [] 时：
    # tools_must 会因「缺工具」判 FAIL（还好），但 tools_forbidden 会判
    # 「一次都没调」= PASS —— 这是 fail-open：没有证据 ≠ 没有违规。
    if not events:
        results.append((
            "trace", False,
            f"轨迹缺失或为空，无法断言工具序列与终止原因：{row.get('trace')}",
        ))
        return results

    if stop:
        # 终止事件取最后一条；不查它会出现「撞满 20 轮被掐断、文件碰巧写对」的假绿
        terminals = [e.get("type") for e in events
                     if e.get("type") in ("final", "max_turns")]
        actual = terminals[-1] if terminals else None
        results.append((
            "stop", actual == stop,
            f"期望终止于 {stop}，实际 {actual or '轨迹里没有任何终止事件（进程被 kill？）'}",
        ))

    if tools_must:
        ok, note = _check_order(tools_must, called)
        results.append(("tools:must", ok, note))

    if tools_forbidden:
        hit = [n for n in called if n in set(tools_forbidden)]
        results.append((
            "tools:forbidden", not hit,
            f"禁调 {tools_forbidden}，实际"
            + (f"调用了 {hit}" if hit else "一次都没调"),
        ))
    return results


@register("receipt", "receipt")
def check_receipt(row, case, expect, events):
    """确认之后实际发生了什么。五条各查一件事，不是把同一件事判五遍。"""
    cards = [e for e in events if e.get("type") == "action_card"]
    receipts = [e for e in events if e.get("type") == "receipt"]
    by_id = {c.get("card_id"): c for c in cards}
    r = receipts[-1] if receipts else None
    paired = by_id.get(r.get("card_id")) if r else None

    return [
        ("receipt:exists", bool(receipts),
         f"轨迹里有 {len(receipts)} 条回执；没有回执 = 没有证据，不是没有违规"),
        ("receipt:source", paired is not None,
         f"回执 card_id {r.get('card_id') if r else None} 必须能追到一张 action_card"
         f"（共 {len(cards)} 张卡）"),
        ("receipt:fingerprint",
         bool(paired) and r.get("fp") == paired.get("fp"),
         f"回执指纹 {r.get('fp') if r else None} / 卡指纹 "
         f"{paired.get('fp') if paired else None}"),
        ("receipt:executed",
         bool(r) and r.get("status") == "executed"
         and isinstance(r.get("after_bytes"), int) and bool(r.get("md5")),
         f"status={r.get('status') if r else None} / "
         f"落盘字节={r.get('after_bytes') if r else None}（必须是实测 int）/ "
         f"md5={r.get('md5') if r else None}"),
        ("receipt:approved_by",
         bool(r) and bool(paired)
         and r.get("approved_by") == paired.get("approved_by"),
         f"回执批准来源 {r.get('approved_by') if r else None} 必须和确认卡一致 "
         f"{paired.get('approved_by') if paired else None} —— 两处对不上说明证据被改过"),
    ]


@register("skill", "skill_injected")
def check_skill(row, case, expect, events):
    """本轮新增的断言类型：验证 Skill（技能）确实被注入、且只注入一次。

    🔴 这一条就是「适配度高」的实证：加它只写了下面 8 行 + 一个装饰器，
    core/evals.py 的 check() 一行都没改。换个 Agent / 换套事件类型，
    照这个契约再加一个 @register 即可。
    """
    want = expect.get("skill_injected")
    names = [e.get("name") for e in events if e.get("type") == "skill"]
    return [
        ("skill:injected", want in names, f"期望注入 {want}，实际 {names}"),
        ("skill:once", len(names) <= 1, f"注入 {len(names)} 次；同一次任务只该注入一次"),
    ]


def run_all(row, case, events=None):
    """按注册表跑全部启用的 checker，返回 [(断言名, bool, 说明), ...]

    🔴 每个 checker 单独兜异常：断言自己炸了要变成一条 FAIL，
    不能让整批评测显示「运行异常」—— 那反而看不出是哪条没过。
    """
    expect = case.get("expect", {})
    if events is None:
        events = read_events(row.get("trace"))
    results = []
    for item in REGISTRY:
        field = item["field"]
        if field is not None and not expect.get(field):
            continue
        try:
            results.extend(item["fn"](row, case, expect, events) or [])
        except Exception as e:
            results.append((
                f"{item['name']}:error", False,
                f"断言自身异常（已转 FAIL，不拖垮整批）：{type(e).__name__}: {e}",
            ))
    return results
