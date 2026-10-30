# -*- coding: utf-8 -*-
"""
扫描笔记语料：找出所有 .md，保留「路径 + 正文」，并统计 篇数 / 总字符数 / 最长一篇

用法：
    python notes_loader.py                    # 直接跑：打印统计
    from notes_loader import load_docs        # 被别人复用：拿到 docs，自己决定怎么用

单篇读取失败不再拖垮整体：跳过 + 记录 + 报告（环境级错误仍然立刻失败）
BOM 从"默默吃掉"变成"吃掉 + 计数可见"
"""
from pathlib import Path


# 语料根目录：写【绝对路径】—— 相对路径的基准是"运行时的工作目录"，换个目录启动就会指向别处
ROOT = Path(r"C:\Users\luotianwen\ai-agent-notes")
# 2026-10-26 库重构（三大章）后改名：07-求职复习 → 91-求职与面试/复习材料；98-代码变更卡 → 02-RAG项目/04-变更卡
# ⚠️ 这两个都能命中【嵌套】目录：is_excluded 比的是全部路径分段，不只是第一层
# 03-变更卡 是约定③之后新建的（Agent 项目的修改记录），与 04-变更卡 同性质：过程产物，不进语料
EXCLUDE_DIRS = {".obsidian", "_附件", "99-原始存档", "复习材料", "04-变更卡", "03-变更卡"}

# 2026-10-27 白名单（本人 11:22 拍板）：语料只取三大章
# 【分工】白名单管「范围」——哪几棵树进来；黑名单管「细节」——树里哪些枝不要
#   例：02-RAG项目 在白名单里，它下面的 04-变更卡 被黑名单剔掉 ⇒ 两者是【叠加】，不是替代
# 🔴 白名单的失效方向是【静默变少】：以后新建一个 04-xxx 目录，整目录笔记不入语料、且不报错
#    ⇒ 所以下面哨兵里必须有「白名单外还有哪些顶层目录」这条输出，否则这个坑查不出来
INCLUDE_TOPS = {"01-初始知识点", "02-RAG项目", "03-Agent项目"}


def is_excluded(p: Path,root:Path=ROOT) -> bool:
    """两段判断：① 顶层目录要在白名单内 ② 路径分段不能命中黑名单

     :param p: 待判断的文件路径（rglob 产出，一定在 root 之下）
     :param root: 语料根目录
     :return: True = 该排除
     """
    dir_parts=p.relative_to(root).parts[:-1] #去掉文件名，只留目录

    # ① 白名单：顶层目录不在名单里 → 整棵子树都不要
    #    散在 root 直下的文件 dir_parts 为空，同样不属于任何一章 ⇒ 一并排除
    if not dir_parts or dir_parts[0] not in INCLUDE_TOPS:
        return True

    # ② 黑名单：范围内再剔掉过程产物（变更卡 / 附件 / 原始存档 / 复习材料）
    return any(part in EXCLUDE_DIRS for part in dir_parts)


def scan_md_files(root: Path=ROOT) -> tuple[list[Path], list[Path]]:
    """扫描 root 下所有 .md，按目录分段过滤

    :return:入选的文件，被排除的文件
    """
    all_md=sorted(p for p in root.rglob("*.md")if p.is_file())
    kept=[p for p in all_md if not is_excluded(p,root)]
    dropped=[p for p in all_md if is_excluded(p,root)]
    return kept, dropped


def load_docs(root: Path = ROOT) -> tuple[list[dict], list[tuple[str, str]], int]:
    """
    扫描并读取 root 下所有 .md —— 【对外接口】，被 basic_rag_pipeline 等复用

    :param root: 语料根目录，默认取本模块的 ROOT
    :return: (docs, errors, bom_files)
             docs      = [{"path": str, "text": str}, ...]  成功读进来的
             errors    = [(路径, 原因), ...]                 跳过没读的
             bom_files = 带 BOM 的文件数（可见性计数）
    """
    # 环境级错误：立刻失败
    if not root.is_dir():
        raise SystemExit(f"语料目录不存在：{root}")

    md_files,dropped = scan_md_files(root)

    # 过滤可见化 + 三条哨兵（任何调用方都会看到）
    tops = sorted({p.relative_to(root).parts[0] for p in dropped})
    print(f"【语料过滤】扫描 {len(md_files) + len(dropped)} 篇 → 入选 {len(md_files)} 篇"
          f"｜排除 {len(dropped)} 篇（{', '.join(tops) if tops else '无'}）")

    all_dir_names = {p.name for p in root.rglob("*") if p.is_dir()}

    # 哨兵①：白名单里每一项，得能在【整棵目录树】里找到同名目录
    # 名字写错 / 目录被改名 ⇒ 整章静默消失，而且篇数看起来"正常变少" —— 这是最贵的假绿
    unknown_inc = INCLUDE_TOPS - all_dir_names
    if unknown_inc:
        print(f"   🔴 白名单里的 {sorted(unknown_inc)} 在整棵目录树里都找不到 —— 整章会被漏掉")

    # 哨兵②：黑名单里每一项，得能找到同名目录（查全树，不只第一层）
    # 2026-10-26 重构把 04-变更卡 挪到了第二层，旧写法（只查 root.iterdir()）会当场误报「名字写错了」
    unknown_exc = EXCLUDE_DIRS - all_dir_names
    if unknown_exc:
        print(f"   ⚠️ 排除名单里的 {sorted(unknown_exc)} 在整棵目录树里都找不到 —— 名字可能写错了")

    # 哨兵③：白名单【外面】还有哪些顶层目录 —— 白名单失效是静默变少，必须每次都摆出来
    md_tops = {p.relative_to(root).parts[0] for p in root.rglob("*.md")
               if len(p.relative_to(root).parts) > 1}
    outside = sorted(md_tops - INCLUDE_TOPS)
    if outside:
        print(f"   ℹ️ 白名单外·整目录不入语料：{', '.join(outside)}")

    if not md_files:
        print("   🔴 白名单非空、却一篇都没选上 —— 过滤彻底失效")

    docs, errors = [], []
    bom_files = 0

    for p in md_files:
        # 先拿字节：能判 BOM，把解码的主动权拿到手
        try:
            raw = p.read_bytes()
        except OSError as e:            # 权限 / 被占用 / 链接断裂 / 读到一半文件没了
            errors.append((str(p), f"{type(e).__name__}: {e}"))
            continue                    # 单文件级：跳过，不拖垮整体

        if raw.startswith(b"\xef\xbb\xbf"):
            bom_files += 1

        # 解码：utf-8-sig 吃掉 BOM；replace 归一化换行
        try:
            text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        except UnicodeDecodeError as e:
            errors.append((str(p), f"不是合法 UTF-8：{e}"))
            continue

        # 空文档过滤：判据是 not text.strip() —— 只有空白/换行也算空（不是 len(text) == 0）
        if not text.strip():
            errors.append((str(p), "空文档：解码后只有空白字符（not text.strip()）"))
            continue

        docs.append({"path": str(p), "text": text})

    return docs, errors, bom_files


def main() -> None:
    """直接运行本文件时：打印统计（复用上面的接口，不重复实现）"""
    docs, errors, bom_files = load_docs()

    total_chars = sum(len(d["text"]) for d in docs)
    longest = max(docs, key=lambda d: len(d["text"])) if docs else None

    print(f"篇数        : {len(docs)}" + (f"（另有 {len(errors)} 篇没吃进去，见下）" if errors else ""))
    print(f"总字符数    : {total_chars}")
    print(f"带 BOM 的文件: {bom_files}")
    if longest is not None:
        print(f"最长的一篇  : {Path(longest['path']).name}（{len(longest['text'])} 字）")
        print(f"  完整路径  : {longest['path']}")
        head = longest["text"][:40].replace("\n", " ").replace("\r", " ")
        print(f"  前 40 字  : {head}")

    # 报出被跳过的：宽容必须配可见，否则就是静默丢数据
    if errors:
        print(f"\n⚠️ 有 {len(errors)} 篇没吃进去 —— 别把它当成 0 篇：")
        for path, why in errors:
            print(f"  - {path}\n      {why}")
        # 想让上游（比如以后的入库流水线）知道"这轮不干净"，就把下面这行的 # 去掉：
        # raise SystemExit(1)


if __name__ == "__main__":
    main()