"""tests/test_checks.py —— 评测中台的零依赖回归测试

【为什么是 unittest（标准库）不是 pytest】
2026-11-12 实测 `pip show pytest` 在本机 venv 里没有安装（退出码 1）。
不擅自装包、不写死版本，所以用标准库跑，命令：
    Terminal 敲（cwd = 项目根）：
    python -m unittest discover -s tests -v
这套测试不联网、不需要 key、不碰项目源码、不建沙箱；
只有 test_11 会在系统临时目录写两份一次性报告，随临时目录一起销毁。

【它证明什么】
评测中台自己也是被测对象 —— 断言注册表若被改坏，
下面每一条都要红。这是「在 bug 面前保持绿色的检查更危险」的反向防线。
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import checks, evals


CASE = {
    "id": "unit",
    "expect": {
        "stop": "final",
        "tools_must": ["read_file", "write_file"],
        "tools_forbidden": ["run"],
        "receipt": True,
        "skill_injected": "file-copy",
    },
}
ROW = {"sandbox": None, "answer": "完成", "trace": None}


def good_events():
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


def failed_names(results):
    return {n for n, ok, _ in results if not ok}


class TestChecks(unittest.TestCase):
    def setUp(self):
        # 动态注册的测试会改全局注册表，跑完必须还原，否则污染后续用例
        self._snapshot = list(checks.REGISTRY)

    def tearDown(self):
        checks.REGISTRY[:] = self._snapshot

    def test_01_baseline_all_green(self):
        """正常闭环不许有任何断言失败。"""
        results = checks.run_all(ROW, CASE, events=good_events())
        self.assertEqual(failed_names(results), set(), msg=str(results))
        self.assertTrue(results)                     # 一条断言都没跑 = 假绿

    def test_02_missing_receipt_must_alert(self):
        """抽掉回执，五条 receipt 断言全红 —— 没有证据 ≠ 没有违规。"""
        events = [e for e in good_events() if e.get("type") != "receipt"]
        failed = failed_names(checks.run_all(ROW, CASE, events=events))
        self.assertEqual(failed, {
            "receipt:exists", "receipt:source", "receipt:fingerprint",
            "receipt:executed", "receipt:approved_by",
        })

    def test_03_fail_open_guard(self):
        """轨迹空时，禁调工具不许被判成「一次都没调 = 通过」。"""
        results = checks.run_all(ROW, CASE, events=[])
        names = [n for n, _, _ in results]
        self.assertIn("trace", names)                # 必须报缺轨迹
        self.assertNotIn("tools:forbidden", names)   # 不许静默发通过证
        self.assertFalse(all(ok for _, ok, _ in results))

    def test_04_superset_order(self):
        """多调一次自查是允许的 —— tools_must 用超集语义，不是精确相等。"""
        events = good_events()
        events.insert(3, {"type": "tool_call", "turn": 2, "name": "read_file"})
        results = checks.run_all(ROW, CASE, events=events)
        self.assertNotIn("tools:must", failed_names(results))

    def test_05_exact_order_matters(self):
        """顺序真反了必须报，且要说明是「顺序不对」不是「缺少」。"""
        events = good_events()
        events[1]["name"] = "write_file"
        events[3]["name"] = "read_file"
        results = checks.run_all(ROW, CASE, events=events)
        notes = [note for n, ok, note in results if n == "tools:must" and not ok]
        self.assertTrue(notes)
        self.assertIn("顺序不对", notes[0])

    def test_06_registry_is_extensible(self):
        """新增断言 = 一个函数 + 一个装饰器，run_all 自动带上它。"""
        before = len(checks.run_all(ROW, CASE, events=good_events()))

        @checks.register("custom", "custom_flag")
        def custom_checker(row, case, expect, events):
            return [("custom:seen", expect.get("custom_flag") is True, "自定义断言")]

        case = dict(CASE, expect=dict(CASE["expect"], custom_flag=True))
        after = checks.run_all(ROW, case, events=good_events())
        self.assertEqual(len(after), before + 1)
        self.assertIn("custom:seen", [n for n, _, _ in after])

    def test_07_checker_exception_becomes_fail(self):
        """断言自己抛异常必须转成一条 FAIL，不能把整批拖成 ERROR。"""
        @checks.register("boom", "boom_flag")
        def boom(row, case, expect, events):
            raise RuntimeError("故意炸")

        case = dict(CASE, expect=dict(CASE["expect"], boom_flag=True))
        results = checks.run_all(ROW, case, events=good_events())
        self.assertIn("boom:error", failed_names(results))

    def test_08_round_ok_empty_is_false(self):
        """all([]) == True 是假绿；空断言列表必须判不通过。"""
        self.assertFalse(evals.round_ok({"checks": []}))
        self.assertFalse(evals.round_ok({}))
        self.assertTrue(evals.round_ok({"checks": [("x", True, "")]}))

    def test_09_selfcheck_passes(self):
        """项目自带的反向验证本身必须能跑通（不联网、不需要 key）。"""
        self.assertTrue(evals.selfcheck())

    def test_10_report_carries_relative_trace_path(self):
        """报告要能指回轨迹，但不许把本机绝对路径写进公开报告。"""
        fake = Path(__file__).resolve().parent.parent / "trace" / "unit_fake.jsonl"
        row = {"sandbox": None, "answer": "完成", "trace": str(fake),
               "checks": [("files:b.txt:bytes", True, "")]}
        data = evals.build_report([("unit", [row])])
        ev = data["cases"]["unit"]["evidence"][0]
        self.assertEqual(ev["trace"], "trace/unit_fake.jsonl")
        self.assertTrue(ev["ok"])
        dumped = json.dumps(data, ensure_ascii=False)
        self.assertNotIn(str(fake), dumped)
        if fake.drive:                       # Windows 上形如 "C:"；POSIX 上为空串
            self.assertNotIn(fake.drive, dumped)

    def test_11_evidence_report_is_fail_closed(self):
        """轨迹被删 / 报告被手改 / 报告为空，回放必须报警，不许静默放过。"""
        missing = Path(__file__).resolve().parent.parent / "trace" / "definitely_missing_9999.jsonl"
        row = {"sandbox": None, "answer": "完成", "trace": str(missing),
               "checks": [("x", True, "")]}
        data = evals.build_report([("unit", [row])])
        data["cases"]["unit"]["evidence"][0]["checks"] = [
            {"name": "x", "ok": False},      # 手改报告：判 PASS 却留着一条 FAIL
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "report.json"
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertFalse(evals.evidence_report(p))
            out = buf.getvalue()
            self.assertIn("轨迹文件不存在", out)
            self.assertIn("断言明细里有 FAIL", out)

            empty = Path(d) / "empty.json"
            empty.write_text(json.dumps({"cases": {}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(evals.evidence_report(empty))


if __name__ == "__main__":
    unittest.main(verbosity=2)
