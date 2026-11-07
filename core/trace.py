"""core/trace.py —— 轨迹层：把 loop 里发生的事逐行落到 JSONL

【职责】
只负责写。不负责读、不负责判定 —— 判定在评测层（core/evals.py，尚未建）。
本模块不 import 任何项目内模块，可以被任何一层调用而不产生循环依赖。

【为什么必须有它】
模型回答的措辞每次不同，不能当回归判据；工具调用序列、参数、返回值、
终止原因才是确定性的。不落盘，这些东西只在终端里活一次，
换一次运行就再也拿不回来，复盘只能靠人肉复制屏幕。

【文件命名】
trace/<YYYY-MM-DD_HHMMSS>.jsonl —— 一次任务一个文件，不同任务不混。
不按会话合并：合并后一次任务的轨迹无法单独取出，评测层还得先切分。

【为什么用 JSONL 而不是一个 JSON 数组】
JSON 必须整份读进内存再解析，写到一半崩溃就整份报废；
JSONL 一行一个事件，写完即落盘，坏掉一行不影响其它行，
且天然适合流式追加 —— 进程被 kill 时已写的行仍然可读。

【格式约定】
每行一个对象，至少含 ts / type。type 当前三种：
  tool_call  模型要调工具   turn / name / args / md5 / len
  final      模型作答       turn / chars
  max_turns  触顶强制退出   turn
新增 type 不需要改本文件 —— 写入层不做白名单校验。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根：core/trace.py 的上一级是 core/，再上一级是 mini-pi/
# 用 __file__ 而不是 cwd —— cwd 会被调用方改，trace 就会散落到任意目录
TRACE_DIR = Path(__file__).resolve().parent.parent / "trace"


def new_trace():
    """新建一个轨迹文件，返回它的路径。一次任务调一次。"""
    TRACE_DIR.mkdir(exist_ok=True)
    return TRACE_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"


def write(trace_path, **event):
    """追加一条事件。成功返回 True，失败返回 False —— 不抛异常。

    🔴 2026-11-05 实测：同一文件第二次追加撞到 PermissionError [Errno 13]
    （trace/ 目录里第一个 .jsonl，被实时扫描短暂锁住）。异常直接冒到 loop.py，
    整次运行崩掉 —— 而模型已经答完、答案也打印完了，进程死在写日志这一步。

    观测层不能搞死被观测对象：轨迹丢了可以再跑一次，答案丢了这一轮就白花。
    所以这里吞掉异常、打一条警告、返回 False，把「记录失败」降级成可忽略的事实。

    newline="" 不可省：默认值会在 Windows 上把 \\n 写成 \\r\\n，
    JSONL 一旦混进 CR，任何按行切分、按字节比对的地方都要额外处理。
    """
    event.setdefault("ts", datetime.now().strftime("%H:%M:%S"))
    line = json.dumps(event, ensure_ascii=False)
    try:
        with open(trace_path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")
        return True
    except Exception as e:
        print(f"⚠️ trace 写入失败（运行继续）：{type(e).__name__}: {e}", file=sys.stderr)
        return False
