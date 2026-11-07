"""core/tools.py —— 工具层：read_file / write_file / run

【职责】
loop 决定是否调用工具，本模块负责执行。向下直接操作文件系统与子进程，
是整个项目唯一会产生不可逆副作用的一层。

【调用链】
模型返回 tool_calls（name + arguments 字符串）
  → run_tool() 校验并 json.loads
  → 分发至 read_file / write_file / run
  → 返回值统一转为 str 交给 loop

【接口】
进：name: str / arguments_json: str（JSON 文本，非 dict）
出：str —— 成功为结果正文；失败为 "错误：" 开头的说明
副作用：写磁盘、启动子进程，仅在模型显式调用时发生

【约定：任何工具都不得向外抛异常】
loop 中以 for tc in msg.tool_calls 遍历并回填 messages。
工具一旦抛异常，该循环立即中断，那条 role="tool" 的消息永远不会写入，
模型收不到失败反馈，表现为程序崩溃而非任务失败。
返回错误字符串后，模型下一轮可读到具体原因并自行改变策略。
普通脚本出错即终止；Agent 出错必须存活，因为错误信息本身是它的输入。

【边界】
不含权限校验、超时控制、输出裁剪、执行决策 —— 分别由 permission 层与调用方负责。
"""

import json
import os
import subprocess
import tempfile


# ============ 工具 Schema ============
# 该结构供模型消费，不供 Python 消费：
# name 用于发起调用，description 决定选用时机，parameters 约束参数填写。

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文本文件的完整内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径，相对于工作目录"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把内容写入文件。文件已存在则覆盖，不存在则创建",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run",
            # description 是模型选工具的唯一依据，模型看不到本机环境。
            # 不写清平台，它会按 Linux 习惯写 pwd / ls / cat（11-05 实测遇到 pwd 失败）。
            "description": (
                "执行一条 shell 命令并返回输出，首行为该命令本身。"
                "当前环境是 Windows，命令解释器为 cmd："
                "列目录用 dir，读文件用 type，删除用 del，删目录用 rd /s /q；"
                "不支持 pwd、ls、cat、rm。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"],
            },
        },
    },
]


# ============ 工具实现 ============
# 三项统一约定：
#   1. 异常一律翻译为「错误：xxx」字符串，不外抛
#   2. 可预判的失败先检查（如 os.path.exists），给出语义化错误而非裸 errno
#   3. 返回值恒为 str，调用方无需做类型判断

def read_file(path):
    """读文件，返回全文。失败时返回错误字符串，不抛异常。"""
    # 预检查：文件不存在是最常见的失败，单独给出可读描述
    # （模型读到「文件不存在」可直接转去列目录，无需猜测原因）
    if not os.path.exists(path):
        return f"错误：文件不存在 {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        # 兜底：文件存在但不可读（权限、编码、路径实际为目录等）
        return f"错误：读取失败 {e}"


def write_file(path, content):
    """写文件，返回一句说明。失败时返回错误字符串，不抛异常。"""
    try:
        # newline="" 关掉换行翻译，内容按字节原样落盘。
        # 默认的 newline=None 会在 Windows 上把每个 \n 改写成 \r\n：
        #   ① 上报的字节数与磁盘实际不符（11-05 实测：报 74，实际 75）
        #   ② 读进来再写出去不再恒等 —— 模型只改一行，整份文件换行符被换掉，
        #      git diff 会显示全文重写。Agent 不得引入模型没要求的改动。
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        # 上报 len(content.encode()) 而非 len(content)：
        # 中文按 UTF-8 编码为 3 字节，字节数才是实际落盘量
        return f"已写入 {len(content.encode('utf-8'))} 字节到 {path}"
    except Exception as e:
        # 典型场景：目标目录不存在（a/b/c.txt 而 a/b 未创建）
        return f"错误：无法写入 {path}：{e}"


def run(command):
    """执行 shell 命令。

    成功：返回 "$ <命令>" + stdout + stderr（首行为命令行回显）
    失败：返回 "错误：命令执行失败：..." 字符串，不抛异常
    """
    try:
        result = subprocess.run(
            command,
            shell=True,  # 模型会写出 "dir /b"、"type a.txt"、"echo x | findstr x" 等 shell 语法
            capture_output=True     # 捕获 stdout 与 stderr；不传 text=True，此处拿到的是 bytes
        )
        # 不使用 text=True：它会按系统默认编码解码，而中文 Windows 的默认编码是 GBK，
        # git / python 等现代工具输出的却是 UTF-8 —— 统一按 GBK 解码必然乱码。
        # 两类输出在真实环境中同时存在，因此取回原始字节后自行分流：
        #   先按 utf-8 严格解码（decode 默认 strict）→ 成功即采用（现代工具走此分支）
        #   抛 UnicodeDecodeError 说明不是 UTF-8，再按 gbk 严格解码（dir / type 等走此分支）
        # 分流判定不可使用 errors="replace"：它会使任意字节都解码成功，
        # utf-8 分支永远命中，gbk 分支形同虚设，等于没有判断。
        out_b, err_b = result.stdout, result.stderr
        try:
            out = out_b.decode("utf-8")
            err = err_b.decode("utf-8")
        except UnicodeDecodeError:
            try:
                out = out_b.decode("gbk")
                err = err_b.decode("gbk")
            except UnicodeDecodeError:
                # 两条严格分支均失败（混合编码或二进制输出）时进入此处。
                # 此时不存在正确结果，errors="replace" 仅用于保底：
                # 保证仍返回字符串（含若干 � 替换符），而不使异常向上逸出。
                out = out_b.decode("utf-8", errors="replace")
                err = err_b.decode("utf-8", errors="replace")
        # 解码结果写回 result，使下一行取到的已是 str，调用方无需再做类型判断
        result.stdout, result.stderr = out, err
        # 命令执行但退出码非 0 不视为异常 —— 那是命令的正常输出（stderr），
        # 模型需要读到它才能自行纠正。真正的异常只有子进程无法启动。
        #
        # 返回值首行回显命令行本身。模型只拿得到输出，看不见自己上一轮写过什么命令；
        # 不回显时它收到的只有「系统找不到指定的路径。」这类不含命令名的文本，
        # 无从判断是哪条失败，只能反复重试（11-05 实测：第 5、6 轮指纹完全相同）。
        # Aider / Claude Code 同样以 "$ " 回显命令。
        return f"$ {command}\n{result.stdout}{result.stderr}"
    except Exception as e:
        # 保持「错误：」开头 —— 与 read_file / write_file 的失败口径一致，
        # 同时附上命令，否则同样无法定位是哪条命令连子进程都没起来。
        return f"错误：命令执行失败：{e}（命令：{command}）"


def run_tool(name, arguments_json):
    """按 name 分发到具体工具。

    name:           "read_file" / "write_file" / "run"
    arguments_json: JSON 字符串，先 json.loads 成字典再使用
    返回：          恒为字符串
    """
    # 第一道校验：arguments 可能不是合法 JSON（模型生成出错）。
    # 同样返回错误字符串而非抛异常 —— 模型下一轮读到解析失败即可重新生成参数。
    try:
        args = json.loads(arguments_json)
    except Exception as e:
        return f"错误：参数不是合法 JSON：{e}"

    # 第二道校验：工具名不存在或必填字段缺失，同样返回错误字符串供模型重试。
    if name == "read_file":
        if "path" not in args:
            return "错误：read_file 缺少参数 path"
        return read_file(args["path"])
    elif name == "write_file":
        if "path" not in args or "content" not in args:
            return "错误：write_file 缺少参数 path 或 content"
        return write_file(args["path"], args["content"])
    elif name == "run":
        if "command" not in args:
            return "错误：run 缺少参数 command"
        return run(args["command"])
    else:
        return f"错误：未知工具 {name}"


# ============ 自测 ============
# 直接运行本文件时执行，用于脱离模型单独验证三个工具。
# 覆盖范围：三条失败路径（文件不存在 / 目录不存在 / 命令不存在），
#           cmd 内置命令（dir / type），中文编码（git 输出），以及分发器校验分支。

if __name__ == "__main__":
    print("--- 成功路径 ---")
    print(write_file("_test_ok.txt", "hello mini-pi"))
    print(read_file("_test_ok.txt"))
    print(run("echo hello from shell"))

    # cmd 内置命令用例：dir / type 不是独立可执行文件，而是 shell 内部命令，
    # 依赖 subprocess 的 shell=True 才能启动 —— 这是 run() 必须保留 shell=True 的直接原因。
    # 同时它们在中文 Windows 上输出 GBK，可用于验证编码分流：
    #   dir /b        仅输出文件名（纯 ASCII），验证「内置命令可执行且能取回输出」
    #   dir（完整版） 输出含中文表头（"的目录"、"个文件"），会真正进入 gbk 分支
    #   type          回读 _test_ok.txt
    # 非 Windows 环境以 ls / cat 替代，保持自测可移植。
    if os.name == "nt":
        print(run("dir /b _test_ok.txt"))    # 期望：_test_ok.txt
        print(run("dir _test_ok.txt"))       # 期望：表头中文正常，非乱码
        print(run("type _test_ok.txt"))      # 期望：hello mini-pi
    else:
        print(run("ls _test_ok.txt"))
        print(run("cat _test_ok.txt"))

    # 中文编码用例：验证 git 的 UTF-8 输出不被解坏 —— 即 run() 弃用 text=True 的直接动因。
    # 直接对当前仓库执行 git log 会依赖「当前目录恰为 git 仓库且最新提交为中文」，
    # 换目录即失效。故此处自建临时仓库并提交一条中文 message，再读回校验。
    #
    # 🔴 仓库建在系统临时目录，不建在项目里。
    # 原写法在 cwd 下建 _test_repo，PyCharm 一看见项目内出现 .git 就把该目录注册成 VCS 根；
    # 自测结束时目录被删，注册记录却留着，于是每次打开项目都弹
    # 「无效 VCS 根映射：core\_test_repo」。自测产物不该污染项目目录，这是通例。
    _tmp_repo = tempfile.mkdtemp(prefix="mini_pi_gittest_")
    run(f'git init -q "{_tmp_repo}"')
    write_file(os.path.join(_tmp_repo, "a.txt"), "a")
    run(f'git -C "{_tmp_repo}" add a.txt')
    # 显式传入 user.name / user.email：机器上可能未配置全局 git 身份，否则 commit 失败
    run(f'git -C "{_tmp_repo}" -c user.name=test -c user.email=test@test commit -q -m "项目版图：core 层三个工具就位"')
    print(run(f'git -C "{_tmp_repo}" log --oneline -1'))  # 期望：中文正常显示

    print("--- 失败路径 1：读不存在的文件 ---")
    print(read_file("no_such_file.txt"))

    print("--- 失败路径 2：往不存在的目录写文件 ---")
    print(write_file("no_such_dir/out.txt", "x"))

    print("--- 失败路径 3：跑不存在的命令 ---")
    print(run("no_such_command_xyz"))

    print("--- 失败路径 4：run_tool 收到非法 JSON / 未知工具 / 缺参数 ---")
    print(run_tool("read_file", "not-a-json"))
    print(run_tool("fly_to_the_moon", "{}"))
    print(run_tool("read_file", "{}"))

    print("--- 失败路径 5：git 仓库不存在时，git log 应返回错误文本而非崩溃 ---")
    print(run("git -C definitely_not_a_repo log --oneline -1"))

    print("--- 失败路径 6：cmd 内置命令读不存在的文件，应返回 stderr 文本而非崩溃 ---")
    if os.name == "nt":
        print(run("type definitely_not_a_file.txt"))
    else:
        print(run("cat definitely_not_a_file.txt"))

    # 清理自测产物（临时 git 仓库在系统临时目录，不在项目里）
    #
    # 🔴 这里不能用 shutil.rmtree(..., ignore_errors=True)：
    # git init 会在 .git/objects 下建只读文件，Windows 上 rmtree 遇到只读文件就失败，
    # 而 ignore_errors=True 让这次失败**完全无声** —— 目录留在 %TEMP% 里，
    # 跑几十次自测就是几十个残留。走 shell 的 rd /s /q 会强制处理只读属性。
    if os.path.exists("_test_ok.txt"):
        os.remove("_test_ok.txt")
    run(f'rd /s /q "{_tmp_repo}"' if os.name == "nt" else f'rm -rf "{_tmp_repo}"')
    print("临时仓库已删：", not os.path.exists(_tmp_repo))
