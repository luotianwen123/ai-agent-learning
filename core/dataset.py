"""core/dataset.py —— 数据层：用例定义 + 沙箱造世界

【职责】
在一次任务**开始之前**准备好世界，之后就不再参与 Agent 循环。
不调用模型、不调用工具、不判定结果 —— 判定在 core/evals.py（未建）。

【为什么这一层必须存在】
Agent 会改文件。如果直接在用例目录里跑：
  第一次跑完，fixture/ 里的 a.txt 被改、b.txt 被创建，
  第二次跑的起点已经不是第一次的起点了。
于是「同一用例跑两次结果不同」永远分不清是模型不稳定还是环境被污染。
沙箱解决的就是这件事：每次跑之前复制一份全新副本，跑完销毁。

【目录约定】
cases/<用例名>/
    case.json      用例定义：任务 + 期望
    fixture/       初始文件（原样复制进沙箱，字节级不改）

【case.json 字段】
  id      str   用例名，与目录名一致
  tags    list  标签，如 ["regression"] / ["smoke"]
  task    str   喂给 run_agent 的任务文本
  expect  dict  期望（本模块不解释它，只原样交给评测层）

【关键约束：跑之前必须切 cwd】
工具里用的是相对路径（"a.txt"），基准是 cwd，不是沙箱路径。
所以评测层在调 run_agent 之前必须 os.chdir(沙箱)，跑完再切回来。
忘记切回来 ⇒ 下一条用例会写进上一条的沙箱，而且不报错。
本模块**不替调用方切 cwd** —— 切换是全局副作用，放在谁调用谁负责。
"""

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

# 项目根：core/dataset.py 的上一级是 core/，再上一级是 mini-pi/
CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


def load_case(case_dir):
    """读一个用例目录，返回 case 字典（额外带 _dir 指向用例目录本身）。"""
    case_dir = Path(case_dir)
    meta_path = case_dir / "case.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"用例缺少 case.json：{case_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["_dir"] = case_dir
    return meta


def list_cases(tag=None):
    """列出 CASES_DIR 下所有合法用例。给了 tag 就只留含该标签的。"""
    if not CASES_DIR.is_dir():
        return []
    out = []
    for d in sorted(CASES_DIR.iterdir()):
        if not d.is_dir() or not (d / "case.json").exists():
            continue
        meta = load_case(d)
        if tag and tag not in meta.get("tags", []):
            continue
        out.append(meta)
    return out


def build_sandbox(case):
    """按该用例的 fixture/ 造一个全新临时目录，返回它的路径。

    用 copytree 而不是逐个读出来再写回去：保字节。
    fixture 里有什么就复制什么 —— 换行符、BOM、空文件都原样带过去。
    昨天 a.txt 被 PowerShell 写成 CRLF 导致指纹变了，就是「字节被二次加工」的例子。
    """
    tmp = Path(tempfile.mkdtemp(prefix="mini_pi_"))
    fixture = Path(case["_dir"]) / "fixture"
    if fixture.is_dir():
        shutil.copytree(fixture, tmp, dirs_exist_ok=True)
    return tmp


def destroy_sandbox(path):
    """销毁沙箱，成功返回 True，失败返回 False 并告警 —— 不静默。

    🔴 2026-11-05 实测：`git init` 会在 .git/objects 下生成**只读**文件。
    Windows 上 rmtree 碰到只读文件直接放弃，再配 ignore_errors=True，
    失败就被吞掉 —— 目录留在 %TEMP% 里没人知道，几周后才发现磁盘少几个 G。
    自测区那次 `_test_repo` 删不掉就是这个原因。

    所以这里用 onerror 回调：遇到删不掉的项，先去掉只读位再重试一次；
    仍然失败就打到 stderr 并返回 False，让调用方看得见。
    """
    path = Path(path)
    if not path.exists():
        return True

    def _force_retry(func, item, exc_info):
        # 去掉只读位后重试；这一步再失败，异常会继续往上抛，不会静默
        os.chmod(item, stat.S_IWRITE)
        func(item)

    try:
        shutil.rmtree(path, onerror=_force_retry)
    except Exception as e:
        print(
            f"⚠️ 沙箱销毁失败，需手动清理：{path}（{type(e).__name__}: {e}）",
            file=sys.stderr,
        )
        return False
    return True
