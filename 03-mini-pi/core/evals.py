"""core/evals.py —— 评测中台：统一入口 / 跑用例 / 落报告 / 前后对比 / 自检

【它最终解决什么】
以前评测只有「终端打印一次结果」，改完代码全凭眼睛看，没法证明没改坏。
现在三件事都能一条命令做完：
    python -m core.evals                     跑全套 + 报告落盘 reports/
    python -m core.evals --selfcheck         故意改坏，断言必须报警（不联网、不需要 key）
    python -m core.evals --compare A.json B.json   两份报告前后对比，发现退化就非零退出
    python -m core.evals --evidence reports/A.json 按报告回放：打印每轮轨迹路径并校验证据
⇒ 「改了能证明没改坏」和「这个 PASS 凭什么」都从口头变成机器判据，CI 能直接接。

【为什么断言不在本文件】
check() 曾经是个 150 行的大函数，四类断言和安全防护全挤在一起。
已全部搬到 core/checks.py 的注册表：新增一类断言 = 加一个 @register 函数，
本文件一行不动。check() 现在只有一行 return。

【边界】
run_one / run_rounds / report / round_ok 的签名与行为保持向后兼容 ——
_scratch 里的桥接脚本、桩驱动仍按原接口调用。
报告目录用 __file__ 定位，不用 cwd（评测会 chdir 到临时沙箱）。
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 直接运行 core/evals.py 时，sys.path[0] 是 core/ 而不是项目根，
# 解释器因此看不见 core 这个包本身，from core import ... 必报 ModuleNotFoundError。
# 这里把项目根补进去；用 `python -m core.evals` 从项目根运行时，这行是空操作，无害。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import checks, dataset, trace

# 向后兼容：外部脚本（_scratch/action_card/bridge.py 等）一直调 evals.read_events
read_events = checks.read_events

# 一条用例跑几次。Agent 是非确定性系统，跑一次等于抛硬币：
# 单次通过率 90% 的用例，连跑 5 次全过只有 59%（0.9⁵）。
# 业界 minimum 是 5（τ-bench 用 5 或 8）；开发调试用 3 省时间，正式报数改 5。
RUNS = 3

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def run_one(case, approver="eval", ask_fn=None):
    """跑一条用例，返回结果字典。

    进：case（就在脚边，不用参数）
        approver 批准策略，评测默认 eval（自动批准且标记 eval:auto）
        ask_fn   模型调用注入点；传桩函数即可离线跑完整主链
    出：{"id": ..., "sandbox": 沙箱路径, "answer": 模型最终回答, "trace": 轨迹文件路径}
    做：建轨迹 → 造沙箱 → 切 cwd 到沙箱 → 调 run_agent(task, trace_path) → 切回 → 返回
    """
    # 延迟 import 放函数内：loop.py 顶层会 load_dotenv() 并拉起 core/llm.py，
    # 摆到模块顶层的话，"只想读用例清单"（dataset.list_cases）也会被拖进去 ——
    # 没配 key 的环境连用例都列不出来。
    # run_agent 在项目根 loop.py，**不在 core/ 包里**，所以是 from loop。
    from loop import run_agent

    # 先建轨迹：trace.new_trace() 内部用的 TRACE_DIR 是绝对路径，
    # 在切 cwd 之前调用，避免任何相对路径歧义。
    trace_path = trace.new_trace()

    # 造沙箱：每个用例一个独立目录，互不污染。
    # 必须传 case —— build_sandbox 要读 case["_dir"] 才找得到该用例的 fixture/
    sandbox = dataset.build_sandbox(case)
    old_cwd = os.getcwd()
    ok = False
    try:
        # 切 cwd：run_agent 里的相对路径读写（ls / 读文件 / 写文件）都落在沙箱内
        os.chdir(sandbox)
        # trace_path 由评测层建好再传进去，run_agent 不再自己 new_trace()，
        # 这样评测层才能拿到轨迹文件路径、与结果一并落进报告
        answer = run_agent(
            case["task"], trace_path=trace_path,
            approver=approver, ask_fn=ask_fn,
        )
        ok = True
        return {
            "id": case.get("id"),
            "sandbox": str(sandbox),
            "answer": answer,
            "trace": str(trace_path),
        }
    finally:
        # 无论成功失败都切回来，否则后续用例的沙箱路径判断会错乱
        os.chdir(old_cwd)
        # 异常路径必须在这里销毁沙箱 —— 别指望外层。
        # run_rounds 的 except 分支是新建一个 row、sandbox 字段填 None，
        # 它手上根本没有沙箱路径，销毁自然跳过。
        # 曾出现 check() 抛异常时沙箱目录被遗留（含写了一半的产物），数小时后才发现。
        # 正常返回时 ok=True，沙箱留给 check 用，由 run_rounds 销毁。
        if not ok:
            dataset.destroy_sandbox(sandbox)


def check(row, case):
    """按 case["expect"] 断言这次运行的事实 —— 交给注册表。

    🔴 本函数刻意只有一行：新增断言类型不许改这里。
    所有防护（fail-open / fail-closed / 超集语义 / 异常兜底）都在 core/checks.py。
    """
    return checks.run_all(row, case)


def round_ok(row):
    """这一轮是否全过。空断言列表不算通过 —— all([]) 返回 True，是假绿。"""
    checks_done = row.get("checks") or []
    return bool(checks_done) and all(ok for _, ok, _ in checks_done)


def run_rounds(case, runs, approver="eval", ask_fn=None):
    """把一条用例跑 runs 次，返回每轮的 row 列表。

    🔴 每轮必须重新造沙箱：沙箱就是为这件事存在的。上一轮模型写出的 b.txt
    会变成下一轮的起点，那样第 2 次跑的根本不是同一条用例 ——
    「同一用例跑三次结果不同」就永远分不清是模型不稳还是环境被污染。
    run_one 内部每次调 dataset.build_sandbox()，天然满足这一点。

    🔴 每轮 check 完立刻销毁沙箱：攒到最后统一销毁的话，
    中间任何一轮异常，前面几轮的沙箱就全泄漏了。
    """
    rounds = []
    for i in range(1, runs + 1):
        try:
            row = run_one(case, approver=approver, ask_fn=ask_fn)
            row["checks"] = check(row, case)
        except Exception as e:
            # 一条用例炸掉不能带走整批：API 超时、限流、网络断在批量评测里是常态
            row = {"id": case.get("id"), "sandbox": None, "answer": None, "trace": None,
                   "round": i}
            row["checks"] = [("run", False, f"运行异常：{type(e).__name__}: {e}")]
        if row.get("sandbox"):
            dataset.destroy_sandbox(row["sandbox"])
        rounds.append(row)
    return rounds


def report(groups):
    """打印每条用例的 N 次结果，以及套件级的单次通过率和 pass^k。

    groups = [(case_id, [row, row, ...]), ...]

    两个指标分开报，因为它们回答的不是同一个问题：
      单次通过率 —— 能力，它「能不能做到」
      pass^k     —— 可靠性，它「是不是每次都行」
    判定（PASS / FAIL）用 pass^k：无人值守跑的时候，要的是每次都行。
    """
    total_rounds = 0
    passed_rounds = 0
    all_pass = 0

    for case_id, rounds in groups:
        n = len(rounds)
        c = sum(1 for r in rounds if round_ok(r))
        total_rounds += n
        passed_rounds += c
        if c == n:
            all_pass += 1

        # 判定用 pass^k 语义：n 次全过才算这条用例过
        print(f"[{'PASS' if c == n else 'FAIL'}] {case_id:<20} {c}/{n}")
        # 只在有失败轮次时打明细。全过时再列 3 轮 × 3 条断言，会把输出淹掉；
        # 失败时必须指到「第几轮、哪条断言、差多少」，否则还得去翻轨迹。
        for i, r in enumerate(rounds, 1):
            if round_ok(r):
                continue
            checks_done = r.get("checks") or []
            print(f"       第 {i} 轮失败：")
            if not checks_done:
                print("         （没有生成任何断言结果）")
            for name, ok, note in checks_done:
                if not ok:
                    print(f"         {name} —— {note}")

    print()
    if total_rounds:
        rate = passed_rounds / total_rounds * 100
        print(f"单次通过率：{passed_rounds}/{total_rounds} 轮（{rate:.1f}%）")
    else:
        print("单次通过率：无数据")
    print(f"pass^{RUNS}：{all_pass}/{len(groups)} 条用例 {RUNS} 次全过")


# ==================== 报告落盘与前后对比 ====================

def build_report(groups):
    """把一次评测压成可存档、可 diff（比对）的 dict（字典）。

    checks_registered 一并记下：两次报告的断言集不同时，通过率不能直接比 ——
    否则「把断言删掉」也能伪装成通过率上升。
    """
    cases = {}
    total_rounds = passed_rounds = all_pass = 0
    for case_id, rounds in groups:
        n = len(rounds)
        c = sum(1 for r in rounds if round_ok(r))
        total_rounds += n
        passed_rounds += c
        if c == n:
            all_pass += 1
        failed = {}
        evidence = []
        for i, r in enumerate(rounds, 1):
            bad = [name for name, ok, _ in (r.get("checks") or []) if not ok]
            if bad:
                failed[str(i)] = bad
            # 报告 → 轨迹的溯源链。只存相对路径：远端是公开仓库，
            # 本机绝对路径进报告就是 11-08 那次安全事件重演。
            # 不存 note（说明文本）：里面可能带沙箱绝对路径或本机路径。
            tp = r.get("trace")
            evidence.append({
                "round": i,
                "ok": round_ok(r),
                "trace": f"trace/{Path(tp).name}" if tp else None,
                "checks": [{"name": name, "ok": bool(ok)}
                           for name, ok, _ in (r.get("checks") or [])],
            })
        cases[case_id] = {
            "rounds": n, "passed": c, "pass_k": c == n, "failed_checks": failed,
            "evidence": evidence,
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": RUNS,
        "cases": cases,
        "total_rounds": total_rounds,
        "passed_rounds": passed_rounds,
        "single_rate": round(passed_rounds / total_rounds * 100, 1)
                       if total_rounds else None,
        "pass_k": f"{all_pass}/{len(groups)}",
        "checks_registered": checks.list_checks(),
    }


def save_report(groups, path=None):
    """报告写进 reports/，返回路径。用 __file__ 定位，不用 cwd。"""
    data = build_report(groups)
    if path is None:
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare_reports(old_path, new_path):
    """两份报告前后对比。发现退化返回 False（命令行据此非零退出，CI 能拦）。

    退化判据三条，任一命中即算退化：
      ① 某条用例从 pass^k 全过变成没全过
      ② 套件单次通过率下降
      ③ 断言集变了 —— 不叫退化，但必须提示「数字不能直接比」
    """
    old = json.loads(Path(old_path).read_text(encoding="utf-8"))
    new = json.loads(Path(new_path).read_text(encoding="utf-8"))
    print(f"旧报告：{Path(old_path).name}")
    print(f"新报告：{Path(new_path).name}")
    print(f"单次通过率：{old.get('single_rate')}% → {new.get('single_rate')}%")
    print(f"pass^{RUNS}：{old.get('pass_k')} → {new.get('pass_k')}")

    regression = False
    for cid in sorted(set(old["cases"]) | set(new["cases"])):
        o, n = old["cases"].get(cid), new["cases"].get(cid)
        if o is None or n is None:
            print(f"  [变化] {cid}：{'新增用例' if o is None else '用例消失'}")
            continue
        if o["pass_k"] and not n["pass_k"]:
            regression = True
            print(f"  [退化] {cid}：{o['passed']}/{o['rounds']} → "
                  f"{n['passed']}/{n['rounds']}，失败断言 {n['failed_checks']}")
        elif n["failed_checks"] != o["failed_checks"]:
            print(f"  [断言变化] {cid}：{o['failed_checks']} → {n['failed_checks']}")
        else:
            print(f"  [无退化] {cid}")

    if old.get("checks_registered") != new.get("checks_registered"):
        print("  ⚠️ 两次注册的断言集不同，通过率不能直接比（删断言也能伪装成涨分）")
    o_rate, n_rate = old.get("single_rate"), new.get("single_rate")
    if o_rate is not None and n_rate is not None and n_rate < o_rate:
        regression = True
    print("结论：" + ("发现退化" if regression else "无退化（改了能证明没改坏）"))
    return not regression


# ==================== 报告 → 证据回放 ====================

def evidence_report(path):
    """按报告把每条用例每一轮的轨迹找出来，打印证据索引并当场校验。

    🔴 这是「有 trace（可回放）」这条卖点的落地点：以前报告只写 PASS/FAIL，
    面试官问「这个 PASS 凭什么」只能重跑一次；现在报告直接指到轨迹文件，
    且当场校验它存在、非空、有终止事件。证据被删、被手改，立刻暴露。

    四条 fail-closed（缺证据判失败）判据：
      ① 报告里没有任何一轮证据       —— 空列表不许算「完整」
      ② 某轮没有 evidence 字段       —— 旧报告不假装能回放
      ③ 轨迹文件不存在 / 为空 / 缺终止事件，而报告判 PASS
      ④ 报告判 PASS，但断言明细里仍有 FAIL（或一条断言都没有）
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    trace_dir = Path(__file__).resolve().parent.parent / "trace"
    print("=" * 66)
    print(f"证据回放：{Path(path).name}")
    print("=" * 66)

    total = 0
    broken = []
    for cid, info in sorted(data.get("cases", {}).items()):
        rounds = info.get("evidence")
        if rounds is None:
            print(f"  [旧报告] {cid} —— 无 evidence 字段，无法回放（请重跑一次生成新报告）")
            broken.append(f"{cid}:旧报告")
            continue
        for item in rounds:
            total += 1
            i = item.get("round")
            verdict = item.get("ok")
            rel = item.get("trace")
            p = trace_dir / Path(rel).name if rel else None
            exists = bool(rel and p and p.is_file())
            events = checks.read_events(p) if exists else []
            seq = [e.get("type") for e in events]
            terminal = [t for t in seq if t in ("final", "max_turns")]
            enough = exists and bool(events) and bool(terminal)

            detail = item.get("checks") or []
            if not exists:
                broken.append(f"{cid}#{i}:轨迹文件不存在")
            elif verdict and not enough:
                broken.append(f"{cid}#{i}:PASS 但轨迹为空或缺终止事件")
            if verdict and not detail:
                broken.append(f"{cid}#{i}:PASS 但没有任何断言明细")
            elif verdict and any(not c.get("ok") for c in detail):
                broken.append(f"{cid}#{i}:报告判 PASS 但断言明细里有 FAIL")

            state = "OK" if exists and (enough or not verdict) else "证据不足"
            print(f"  [{state}] {cid} 第 {i} 轮 {'PASS' if verdict else 'FAIL'} "
                  f"→ {rel or '无轨迹'} ｜ {len(events)} 个事件："
                  f"{' → '.join(seq) or '空'}")

    if total == 0:
        broken.append("报告里没有任何一轮证据（空列表不许算完整）")
    print(f"共 {total} 轮；证据链完整 = {not broken}")
    if broken:
        print("断点：" + "；".join(broken))
    return not broken


# ==================== 断言层自检（不需要模型 / 网络 / key） ====================

_SELFCHECK_CASE = {
    "id": "selfcheck",
    "expect": {
        "stop": "final",
        "tools_must": ["read_file", "write_file"],
        "tools_forbidden": ["run"],
        "receipt": True,
        "skill_injected": "file-copy",
    },
}


def _good_events():
    """一份「正常闭环」的轨迹事件，纯内存构造，不碰磁盘。"""
    return [
        {"type": "skill", "turn": 0, "name": "file-copy"},
        {"type": "tool_call", "turn": 1, "name": "read_file"},
        {"type": "action_card", "turn": 2, "card_id": "card-x", "name": "write_file",
         "fp": "abc123456789", "approved": True, "approved_by": "eval:auto"},
        {"type": "tool_call", "turn": 2, "name": "write_file"},
        {"type": "receipt", "turn": 2, "card_id": "card-x", "fp": "abc123456789",
         "status": "executed", "after_bytes": 74, "md5": "deadbeef0000",
         "approved_by": "eval:auto"},
        {"type": "final", "turn": 3},
    ]


def _mutate(fn):
    events = _good_events()
    fn(events)
    return events


def selfcheck():
    """故意改坏八处，断言必须各自报警。全报警才返回 True。

    这是「反向验证」内建进项目：以前它躺在 _scratch 脚本里，
    别人 git clone 下来根本看不见；现在一条命令就能证明断言不是摆设。
    """
    print("=" * 66)
    print("评测中台自检 · 故意改坏，断言必须报警（不报警 = 假测试）")
    print(f"当前注册断言器：{[c['name'] for c in checks.list_checks()]}")
    print("=" * 66)

    row = {"sandbox": None, "answer": "已完成", "trace": None}
    scenarios = [
        ("0 正常样本（必须全绿）", _good_events(), set(), True),
        ("1 抽掉回执", _mutate(lambda e: e.remove(e[4])),
         {"receipt:exists", "receipt:source", "receipt:fingerprint",
          "receipt:executed", "receipt:approved_by"}, False),
        ("2 回执指纹被改", _mutate(lambda e: e[4].update(fp="ffffffffffff")),
         {"receipt:fingerprint"}, False),
        ("3 自动批准冒充人批准", _mutate(lambda e: e[4].update(approved_by="human")),
         {"receipt:approved_by"}, False),
        ("4 落盘字节数不是实测值", _mutate(lambda e: e[4].update(after_bytes=None)),
         {"receipt:executed"}, False),
        ("5 调了禁用工具 run", _mutate(
            lambda e: e.insert(3, {"type": "tool_call", "turn": 2, "name": "run"})),
         {"tools:forbidden"}, False),
        ("6 工具调用顺序反了", _mutate(lambda e: e[1].update(name="write_file")),
         {"tools:must"}, False),
        ("7 没注入 Skill", _mutate(lambda e: e.remove(e[0])),
         {"skill:injected"}, False),
        ("8 轨迹整个空掉", [], {"trace"}, False),
    ]

    all_ok = True
    for label, events, must_fail, baseline in scenarios:
        results = checks.run_all(row, _SELFCHECK_CASE, events=events)
        failed = {n for n, ok, _ in results if not ok}
        ok = (not failed) if baseline else must_fail.issubset(failed)
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} —— 报警 {sorted(failed) or '无'}")

    # 单独防 fail-open：轨迹空时，禁调工具不许被判成「一次都没调 = 通过」
    empty = checks.run_all(row, _SELFCHECK_CASE, events=[])
    names = [n for n, _, _ in empty]
    leak = "tools:forbidden" in names
    print(f"  [{'FAIL' if leak else 'PASS'}] 9 轨迹空时禁调工具不许静默通过"
          f" —— 产出断言 {names}")
    all_ok = all_ok and not leak

    print("结论：" + ("自检通过，断言确实在拦" if all_ok else "自检失败，存在假测试"))
    return all_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="mini-pi 评测中台")
    p.add_argument("--selfcheck", action="store_true",
                   help="断言层反向验证；不联网、不需要 key")
    p.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"),
                   help="对比两份已落盘的报告，发现退化则非零退出")
    p.add_argument("--evidence", metavar="REPORT",
                   help="按报告回放证据：打印每轮轨迹路径与事件序列，缺证据则非零退出")
    p.add_argument("--out", default=None, help="报告输出路径（缺省自动写 reports/）")
    args = p.parse_args()

    if args.selfcheck:
        sys.exit(0 if selfcheck() else 1)
    if args.compare:
        sys.exit(0 if compare_reports(*args.compare) else 1)
    if args.evidence:
        sys.exit(0 if evidence_report(args.evidence) else 1)

    # 顺序是硬约束：每轮 run_one → check → destroy，三轮各自走一遍。
    groups = []
    cases = dataset.list_cases()
    if not cases:
        print("cases/ 下没有找到任何用例")
    for case in cases:
        print(f"\n=== {case.get('id')}：跑 {RUNS} 次 ===")
        groups.append((case.get("id"), run_rounds(case, RUNS)))
    report(groups)
    if groups:
        print(f"报告已落盘：{save_report(groups, args.out)}")
