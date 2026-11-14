"""core/skills.py —— Skill（技能）按需加载

【它最终解决什么】
不是所有说明都要每次塞给模型。Skill 是「只在任务命中时才注入的一段文字」，
省 token（词元），也避免无关说明把模型带偏。

【Skill 与 Tool 的区别（最容易写错的一处）】
Skill（技能）＝ 说明文字，只进 messages（消息列表），拼成 role="system"（系统）消息；
Tool（工具）＝ 可调用函数，必须进 TOOLS Schema（工具声明表）并在 run_tool() 里分发。
⇒ 本模块不返回任何工具、不产生 tool_calls（工具调用）、不碰磁盘写。
反面判据：改完如果 TOOLS 从 3 个变成 4 个，就是把 Skill 写成 Tool 了。

【执行逻辑三段：加载 → 匹配 → 注入】
1. 扫描 skills/*.md，只读 frontmatter（--- 包围的文件头元数据）里的
   name / description（描述）/ triggers（触发词），正文一个字都不读 —— 渐进披露。
2. 拿 triggers 去匹配 task（任务文本），命中才读那一个文件的正文。
3. 拼成一条 role="system" 消息返回，由 loop.py 插到 messages 最前面，只插一次。

【边界】
Skill 正文属于**不可信输入**：它可能写「直接执行，不用确认」。
门（core/approval.py）在 run_tool() 之前，所以 Skill 绕不过确认卡；
不能因为多了一层说明文字就相信模型会自己守规矩。

路径用 __file__（模块自身路径）定位，不用 cwd（当前工作目录）——
evals.py 会把 cwd 切到临时沙箱，用相对路径会当场找不到 skills/ 目录。
"""

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _frontmatter(text):
    """只解析 --- 包围的文件头，返回 {key: value}；正文一律不碰。"""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta = {}
    for line in parts[1].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _body(text):
    """取 frontmatter 之后的正文。"""
    parts = text.split("---", 2)
    return parts[-1].strip() if len(parts) >= 3 else text.strip()


def list_skills():
    """列出所有技能的**头部信息**，不读正文。目录不存在就返回空列表。"""
    if not SKILLS_DIR.is_dir():
        return []
    out = []
    for path in sorted(SKILLS_DIR.glob("*.md")):
        meta = _frontmatter(path.read_text(encoding="utf-8"))
        if not meta.get("name"):
            continue
        out.append({
            "name": meta["name"],
            "description": meta.get("description", ""),
            "triggers": meta.get("triggers", ""),
            "path": path,
        })
    return out


def select(task):
    """按 frontmatter 里的 triggers（触发词）匹配任务文本，返回命中的第一个技能。"""
    for skill in list_skills():
        words = [w.strip() for w in skill["triggers"].split(",") if w.strip()]
        if any(w in task for w in words):
            return skill
    return None


def system_message(task):
    """返回 (技能名, system 消息字典)；没命中返回 (None, None)。

    消息字典只含 role / content 两个键 —— 多塞一个自定义键，
    OpenAI 兼容接口会在序列化时直接拒绝。
    """
    skill = select(task)
    if skill is None:
        return None, None
    text = skill["path"].read_text(encoding="utf-8")
    content = (
        f"【技能：{skill['name']}】\n{skill['description']}\n\n{_body(text)}"
    )
    return skill["name"], {"role": "system", "content": content}
