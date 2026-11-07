"""core/evals.py —— 评测层：跑用例 → 断言 → 报告"""

import os
import sys
from pathlib import Path

# 直接运行 core/evals.py 时，sys.path[0] 是 core/ 而不是项目根，
# 解释器因此看不见 core 这个包本身，from core import ... 必报 ModuleNotFoundError。
# 这里把项目根补进去；用 `python -m core.evals` 从项目根运行时，这行是空操作，无害。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import dataset, trace


def run_one(case):
    """跑一条用例，返回结果字典。

    进：case（就在脚边，不用参数）
    出：{"id": ..., "sandbox": 沙箱路径, "answer": 模型最终回答, "trace": 轨迹文件路径}
    做：造沙箱 → 切 cwd 到沙箱 → 调 run_agent(case["task"]) → 切回 → 返回
    """
    # 延迟 import 放函数内：loop.py 顶层会 load_dotenv() 并拉起 core/llm.py，
    # 摆到模块顶层的话，"只想读用例清单"（dataset.list_cases）也会被拖进去 ——
    # 没配 key 的环境连用例都列不出来。
    # 🔴 run_agent 在项目根 loop.py，**不在 core/ 包里**，所以是 from loop。
    from loop import run_agent

    # 造沙箱：每个用例一个独立目录，互不污染。
    # 必须传 case —— build_sandbox 要读 case["_dir"] 才找得到该用例的 fixture/
    sandbox = dataset.build_sandbox(case)
    old_cwd = os.getcwd()
    try:
        # 切 cwd：run_agent 里的相对路径读写（ls / 读文件 / 写文件）都落在沙箱内
        os.chdir(sandbox)
        answer = run_agent(case["task"])
    finally:
        # 无论成功失败都切回来，否则后续用例的沙箱路径判断会错乱
        os.chdir(old_cwd)

    # 🔴 轨迹路径暂时拿不到：run_agent 内部自己 trace.new_trace()，但不把路径返回。
    # 下一步给 run_agent 加可选参数 trace_path=None（loop.py 第 44/53 行）之后再接上。
    # 这里先占位 None —— run_one 的返回契约（四个 key）保持不变，评测层暂不依赖 trace。
    return {
        "id": case.get("id"),
        "sandbox": str(sandbox),
        "answer": answer,
        "trace": None,
    }


def check(expect, sandbox):
    """按 case["expect"] 断言，返回 [(断言名, True/False, 说明), ...]

    本轮只做一条断言就够：expect["files"] 里每个文件在 sandbox 下是否存在。
    """
    results = []
    sandbox = Path(sandbox)
    for item in expect.get("files", []):
        path = item.get("path", "")
        want = bool(item.get("exists", True))        # 缺省按「期望存在」处理
        actual = (sandbox / path).exists()
        # 拼的是「沙箱绝对路径 + 相对文件名」，与当前 cwd 无关 ——
        # 项目根目录下恰好也有一份 a.txt / b.txt，一旦用相对路径就会查错地方
        ok = actual == want
        note = f"期望{'存在' if want else '不存在'}，实际{'存在' if actual else '不存在'}"
        results.append((f"files:{path}", ok, note))
    return results


def report(rows):
    """逐条打印 PASS / FAIL。"""
    passed = 0
    for row in rows:
        checks = row.get("checks", [])
        if not checks:
            print(f"[????] {row.get('id')}：没有生成任何断言结果")
            continue
        all_ok = all(ok for _, ok, _ in checks)
        if all_ok:
            passed += 1
        print(f"[{'PASS' if all_ok else 'FAIL'}] {row.get('id')}")
        for name, ok, note in checks:
            if not ok:
                print(f"       {name} —— {note}")
    print(f"\n合计：{passed}/{len(rows)} 通过")


if __name__ == "__main__":
    # 顺序是硬约束：run_one → check → report → destroy。
    # check 必须发生在 destroy 之前，否则沙箱已删、所有 exists 恒为 False、必 FAIL。
    rows = []
    cases = dataset.list_cases()
    if not cases:
        print("cases/ 下没有找到任何用例")
    for case in cases:
        try:
            row = run_one(case)
            row["checks"] = check(case.get("expect", {}), row["sandbox"])
        except Exception as e:
            # 一条用例炸掉不能带走整批：API 超时、网络断都常见。
            # 标记 ERROR 并补一条必败断言，让它出现在报告里而不是让进程死掉
            row = {"id": case.get("id"), "sandbox": None, "answer": None, "trace": None}
            row["checks"] = [("run", False, f"运行异常：{type(e).__name__}: {e}")]
        rows.append(row)
    report(rows)
    for row in rows:
        if row.get("sandbox"):
            dataset.destroy_sandbox(row["sandbox"])
