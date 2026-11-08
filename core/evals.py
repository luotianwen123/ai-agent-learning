"""core/evals.py —— 评测层：跑用例 → 断言 → 报告"""

import json
import os
import sys
from pathlib import Path

# 直接运行 core/evals.py 时，sys.path[0] 是 core/ 而不是项目根，
# 解释器因此看不见 core 这个包本身，from core import ... 必报 ModuleNotFoundError。
# 这里把项目根补进去；用 `python -m core.evals` 从项目根运行时，这行是空操作，无害。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import dataset, trace

# 一条用例跑几次。Agent 是非确定性系统，跑一次等于抛硬币：
# 单次通过率 90% 的用例，连跑 5 次全过只有 59%（0.9⁵）。
# 业界 minimum 是 5（τ-bench 用 5 或 8）；开发调试用 3 省时间，正式报数改 5。
RUNS = 3


def run_one(case):
    """跑一条用例，返回结果字典。

    进：case（就在脚边，不用参数）
    出：{"id": ..., "sandbox": 沙箱路径, "answer": 模型最终回答, "trace": 轨迹文件路径}
    做：建轨迹 → 造沙箱 → 切 cwd 到沙箱 → 调 run_agent(task, trace_path) → 切回 → 返回
    """
    # 延迟 import 放函数内：loop.py 顶层会 load_dotenv() 并拉起 core/llm.py，
    # 摆到模块顶层的话，"只想读用例清单"（dataset.list_cases）也会被拖进去 ——
    # 没配 key 的环境连用例都列不出来。
    # 🔴 run_agent 在项目根 loop.py，**不在 core/ 包里**，所以是 from loop。
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
        answer = run_agent(case["task"], trace_path=trace_path)
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
        # 🔴 异常路径必须在这里销毁沙箱 —— 别指望外层。
        # run_rounds 的 except 分支是新建一个 row、sandbox 字段填 None，
        # 它手上根本没有沙箱路径，销毁自然跳过。
        # %TEMP%\mini_pi_8ym4lfp9 就是这么漏的：10:26 那次 check() 抛 TypeError，
        # 目录带着写了一半的 b.txt 一直躺到 11:13 才被发现。
        # 正常返回时 ok=True，沙箱留给 check 用，由 run_rounds 销毁。
        if not ok:
            dataset.destroy_sandbox(sandbox)


def read_events(trace_path):
    """读 JSONL 轨迹，返回事件列表。读不到、坏行，一律跳过 —— 不抛异常。

    同一条原则：断言函数不许抛。trace 正被占用、进程被 kill 时文件可能写到一半，
    但那不该把整条用例变成 ERROR —— 拿不到轨迹就断言「轨迹里没有禁止调用」，
    该报 FAIL 就报 FAIL，而不是让整个进程死掉。
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
    """need 是否按**顺序**出现在 called 里，允许中间插别的调用。返回 (ok, 说明)。

    这是 superset（超集）语义：只要求该做的一个不漏、先后不乱，
    不要求 called 和 need 长度相等。业内（Anthropic《Demystifying evals for
    AI agents》）明确反对精确相等 —— 那个太 rigid，会把「多调一次自查」
    这种好行为判成失败。
    """
    pos = 0
    for name in need:
        try:
            pos = called.index(name, pos) + 1
        except ValueError:
            # 失败必须说清是**哪一种**：没调 vs 调了但顺序不对。
            # 两者修法完全不同 —— 一个是模型不知道该调什么，一个是它知道但先做了后一步。
            # 报告只写「不匹配」，人还得自己去翻轨迹，那这条断言的价值就少了一半。
            if name in called:
                return False, f"顺序不对：{name} 出现在它该出现的位置之前 —— 实际 {called}"
            return False, f"缺少 {name} —— 实际只调用了 {called}"
    return True, f"必须按序包含 {need}，实际 {called}"


def check(row, case):
    """按 case["expect"] 断言这次运行的事实，返回 [(断言名, True/False, 说明), ...]

    进：
      row  —— run_one 产出的事实：sandbox / answer / trace
      case —— 用例定义：expect / _dir
    出：断言三元组列表，交给 report 打印

    🔴 断言函数绝不能抛异常：模型什么都没写，是最该被测的情况，
    这时 (sandbox / path).read_bytes() 会抛 FileNotFoundError，
    整条用例变成 ERROR —— 报告只显示「运行异常」，
    看不出到底哪条断言没过。所以先判存在，再判内容。
    """
    expect = case.get("expect", {})
    results = []

    # ---------------- 文件类断言：查环境终态 ----------------
    # 业内排序上这是第一层（outcome），比路径断言重要：
    # agent 说「已写入 b.txt」不算数，磁盘上真有且字节对才算数。
    sandbox = Path(row["sandbox"]) if row.get("sandbox") else None
    if sandbox is not None:
        for item in expect.get("files", []):
            path = item.get("path", "")
            # 拼的是「沙箱绝对路径 + 相对文件名」，与当前 cwd 无关 ——
            # 项目根目录下恰好也有一份 a.txt / b.txt，一旦用相对路径就会查错地方
            target = sandbox / path

            # ---------- 断言 1：存在性 ----------
            want = bool(item.get("exists", True))          # 缺省按「期望存在」处理
            actual_exists = target.exists()
            results.append((
                f"files:{path}:exists",
                actual_exists == want,
                f"期望{'存在' if want else '不存在'}，实际{'存在' if actual_exists else '不存在'}",
            ))
            # 文件不在，就没内容可比；期望它不存在时也不该继续比
            if not actual_exists or not want:
                continue

            # ---------- 断言 2：内容逐字节相同（可选） ----------
            # 只有 case.json 里写了 same_bytes_as 才做。
            # 用 .get()：用例 JSON 是人手改的文件，缺字段不能把整批拖崩
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

    # ---------------- 工具序列断言：查路径上的不变量 ----------------
    # 业内做法不是比精确序列，而是只断言两类不变量：
    #   ① tools_must      该做的一个不许漏、先后不许乱（允许额外调用）
    #   ② tools_forbidden 不该做的一次都不许出现
    # 依据：同一条用例两次运行，模型一次多调了 `run type b.txt` 自查 ——
    # 那是好行为，精确相等会把它判成失败，测试立刻变 brittle。
    tools_must = expect.get("tools_must") or []
    tools_forbidden = expect.get("tools_forbidden") or []
    if tools_must or tools_forbidden:
        trace_path = row.get("trace")
        events = read_events(trace_path)
        called = [e.get("name") for e in events if e.get("type") == "tool_call"]

        # 🔴 轨迹缺失必须单独报 FAIL，不能让它悄悄溜过去。
        # 否则 called = [] 时：tools_must 会因为「缺工具」判 FAIL（fail-closed，还好），
        # 但 tools_forbidden 会因为「一次都没调」判 PASS —— 这是 fail-open：
        # 没有证据 ≠ 没有违规。删掉那条读取日志，就等于给自己发了一张无违规证明。
        if not events:
            results.append((
                "tools:trace",
                False,
                f"轨迹缺失或为空，无法断言工具序列：{trace_path}",
            ))
        else:
            if tools_must:
                ok, note = _check_order(tools_must, called)
                results.append(("tools:must", ok, note))

            if tools_forbidden:
                hit = [n for n in called if n in set(tools_forbidden)]
                results.append((
                    "tools:forbidden",
                    not hit,
                    f"禁调 {tools_forbidden}，实际"
                    + (f"调用了 {hit}" if hit else "一次都没调"),
                ))
    return results


def round_ok(row):
    """这一轮是否全过。空断言列表不算通过 —— all([]) 返回 True，是假绿。"""
    checks = row.get("checks") or []
    return bool(checks) and all(ok for _, ok, _ in checks)


def run_rounds(case, runs):
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
            row = run_one(case)
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
            checks = r.get("checks") or []
            print(f"       第 {i} 轮失败：")
            if not checks:
                print("         （没有生成任何断言结果）")
            for name, ok, note in checks:
                if not ok:
                    print(f"         {name} —— {note}")

    print()
    if total_rounds:
        rate = passed_rounds / total_rounds * 100
        print(f"单次通过率：{passed_rounds}/{total_rounds} 轮（{rate:.1f}%）")
    else:
        print("单次通过率：无数据")
    print(f"pass^{RUNS}：{all_pass}/{len(groups)} 条用例 {RUNS} 次全过")


if __name__ == "__main__":
    # 顺序是硬约束：每轮 run_one → check → destroy，三轮各自走一遍。
    groups = []
    cases = dataset.list_cases()
    if not cases:
        print("cases/ 下没有找到任何用例")
    for case in cases:
        print(f"\n=== {case.get('id')}：跑 {RUNS} 次 ===")
        groups.append((case.get("id"), run_rounds(case, RUNS)))
    report(groups)
