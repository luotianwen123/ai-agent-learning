"""core/trace.py —— 轨迹层：把 loop 里发生的事逐行落到 JSONL

【职责】
只负责写。不负责读、不负责判定 —— 判定在评测层（core/evals.py，尚未建）。
本模块不 import 任何项目内模块，可以被任何一层调用而不产生循环依赖。

【为什么必须有它】
模型回答的措辞每次不同，不能当回归判据；工具调用序列、参数、返回值、
终止原因才是确定性的。不落盘，这些东西只在终端里活一次，
换一次运行就再也拿不回来，复盘只能靠人肉复制屏幕。

【文件命名】
trace/<YYYY-MM-DD_HHMMSS_ffffff>.jsonl —— 一次任务一个文件，不同任务不混。
带微秒（ffffff）不是装饰：只到秒时，同一秒内建的两个轨迹会写进同一个文件，
两条任务的事件交错落在一起，评测阶段就分不清哪条证据属于谁。
不按会话合并：合并后一次任务的轨迹无法单独取出，评测层还得先切分。

【为什么用 JSONL 而不是一个 JSON 数组】
JSON 必须整份读进内存再解析，写到一半崩溃就整份报废；
JSONL 一行一个事件，写完即落盘，坏掉一行不影响其它行，
且天然适合流式追加 —— 进程被 kill 时已写的行仍然可读。

【格式约定】
每行一个对象，至少含 ts / type。type 当前三种：
  tool_call  模型要调工具   turn / name / args / md5 / len / result / truncated
  final      模型作答       turn / chars
  max_turns  触顶强制退出   turn
新增 type 不需要改本文件 —— 写入层不做白名单校验。

【为什么 tool_call 要带 result（2026-11-08 补）】
只有 md5 和 len 的轨迹，回放时只能证明「工具被调过、返回了 1234 个字符」，
但返回的到底是什么，看不出来。出了故障只能靠人肉回忆终端屏幕。
md5 留着的用途变了：不再是「唯一线索」，而是「两次运行返回是否逐字节一致」
的校验值 —— 配合 result 一起看，才知道差异出在哪一段。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根：core/trace.py 的上一级是 core/，再上一级是 mini-pi/
# 用 __file__ 而不是 cwd —— cwd 会被调用方改，trace 就会散落到任意目录
TRACE_DIR = Path(__file__).resolve().parent.parent / "trace"


def new_trace():
    """新建一个轨迹文件，返回它的路径。一次任务调一次。

    🔴 文件名必须带毫秒（%f）—— 只用 %H%M%S 时，同一秒内建的两次轨迹
    会落到**同一个文件**上：两条任务的 tool_call / final 事件交错写进一个 JSONL，
    评测层再按文件名去取证据，就变成「拿 A 的轨迹去证明 B 通过」。
    这类错不报错、数字还好看，属于最坏的一种。

    这不是假设出来的风险 —— trace/ 里同时躺着两种命名：
    2026-11-07_103849_655440.jsonl（带毫秒）和 2026-11-07_100201.jsonl（不带），
    说明带毫秒的版本曾经有过，后来被改回去了。这次补回来。

    再退一步：连微秒都撞上（多进程同刻启动），就加 -2 / -3 后缀。
    唯一性不能建立在「应该不会撞」上 —— 撞了不报错，就是静默污染证据。

    🔴 第一版这里写的是 `while path.exists(): path = 加后缀` —— 那是**假兜底**。
    实测：20 线程 × 50 次并发建轨迹，1000 次只得到 265 个不同文件名，
    重复 735 次。原因是 exists() 只有在文件**已经存在**时才为真，
    而本函数原先只返回路径、不建文件：同一微秒的两次调用都看到「不存在」，
    于是都返回同一个路径 —— 检查看着有，实际一次都没生效。

    修法：这里真的把文件建出来，用 "x" 模式（独占创建，已存在就抛
    FileExistsError）。唯一性由文件系统保证，是原子的 ——
    不靠内存里的一个 set，也不靠「检查完到使用之间没人插队」这种假设。
    代价是任务异常时会留一个空轨迹文件；那本身就是证据：
    任务开始了，但一条记录都没落下来。
    """
    TRACE_DIR.mkdir(exist_ok=True)
    base = f"{datetime.now():%Y-%m-%d_%H%M%S_%f}"
    for n in range(1, 101):
        path = TRACE_DIR / (f"{base}.jsonl" if n == 1 else f"{base}-{n}.jsonl")
        try:
            with open(path, "x", encoding="utf-8"):
                pass                      # 独占占住这个名字，空文件即可
            return path
        except FileExistsError:
            continue                      # 撞了，换下一个后缀
    raise RuntimeError(
        f"同一个微秒内建了 100 个轨迹文件仍未拿到空闲文件名：{base}.jsonl "
        f"—— 要么时钟被钉住，要么并发远超设计假设，需人工看。"
    )


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


# 截断阈值的唯一来源：brief() 用它做默认值，loop.py 用它判断 truncated 标记。
# 两处若各写一份数字，改了一处另一处不跟，轨迹里就会出现「标着未截断、实际被截断」的假信息。
BRIEF_HEAD = 400
BRIEF_TAIL = 200


def brief(text, head=BRIEF_HEAD, tail=BRIEF_TAIL):
    """把工具返回值压成「可回放摘要」：留头 head 字符 + 尾 tail 字符，中间省略。

    为什么必须截断：read_file 读一个几百行的文件、run 跑一条报错的命令，
    返回值动辄上万字符。原样塞进轨迹，一个用例就能把 JSONL 撑到几十 KB，
    而复盘时真正要看的往往只有两头。

    为什么是两头而不是只留头：
        错误信息在尾部 —— traceback 最后一行才是异常类型（KeyError / Timeout），
        只留头 = 看到一堆调用栈，看不到到底报了什么错；
        上下文在头部 —— 文件第一行、命令的回显、JSON 的起始结构，
        只留尾 = 看到一堆收尾括号，不知道这是哪个文件的内容。
    只留一头，必然有一半故障看不出来。

    截断到什么程度还算「可回放」：调了什么（name）、参数是什么（args）、
    返回的开头和结尾、原文总长（事件里的 len）。要看全文就重跑一次 ——
    轨迹的定位是「够定位问题」，不是「替代原始数据」。
    """
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...（省略 {omitted} 字符）...\n{text[-tail:]}"
