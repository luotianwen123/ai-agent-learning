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

    md_files = sorted(p for p in root.rglob("*.md") if p.is_file())

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