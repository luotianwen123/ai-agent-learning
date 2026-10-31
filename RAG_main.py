# -*- coding: utf-8 -*-
"""RAG_main.py —— 一条命令问一个问题

    python RAG_main.py "什么是RAG？"
    不带参数则进交互模式，模型只加载一次。

库不在就自己建（约 60 秒），所以别人 clone 下来不用先跑 chroma_build.py。
检索先看距离：太远就不让模型答，免得它编。
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()   # 必须在 basic_rag_pipeline 之前：它在模块级读 OPENAI_API_KEY

import sys
from pathlib import Path

import tiktoken
from sentence_transformers import SentenceTransformer

from notes_loader import load_docs
from basic_rag_pipeline import (
    build_corpus,
    calc_available_chunk_quota,
    clip_context_by_max_token,
    build_rag_prompt,
    llm_chat,
    BASE_URL,
    MODEL,
)
from chroma_store import (
    DB_PATH,
    COLLECTION,
    MAX_CHUNK_SIZE,
    OVERLAP,
    _get_collection,
    build_index,
    search,
    search_as_chunks,
)

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 5
MODEL_MAX_WINDOW = 8192
RESERVE_OUTPUT_TOKENS = 512
TOKENIZER_NAME = "cl100k_base"

# 拒答阈值：2026-10-31 标定（47 篇 / 2713 块，10 道正样本 + 20 个无关问题）
#   正样本（正确块）最大距离 0.3278 → 比它小算强相关
#   负样本（无关问题 top1）最小距离 0.4105 → 比它大直接拒答
# 语料一变这两个数就得重标，脚本在 工作台\RAG阈值标定\阈值标定.py
DIST_STRONG = 0.3278
DIST_REJECT = 0.4105

SYSTEM_PROMPT = (
    "你是知识库问答助手，只依据【参考文档】回答。"
    "文档里没有依据时，直接回答「知识库中没有相关内容」，不要编造。"
)


def ensure_index(model) -> None:
    """库能用就直接走，不能用就重建。"""
    # 目录在、但 collection 缺失或索引坏了，也算不可用 → 重建
    if DB_PATH.exists():
        try:
            if _get_collection(DB_PATH, COLLECTION).count() > 0:
                return
        except Exception:
            pass

    # 建库要 60 秒左右，先说一声，不然会被当成卡死
    print("首次运行，正在建库（约 60 秒）……")

    docs, errors, bom = load_docs()
    if errors:
        print(f"  {len(errors)} 篇没读进来（不影响建库，只是语料少了点）")

    all_positions, report, failures = build_corpus(docs, MAX_CHUNK_SIZE, OVERLAP)
    # 有失败就停：残缺的库比没有库更难查
    assert not failures, f"{len(failures)} 篇没通过校验，拒绝建残缺库"

    vectors = model.encode(
        [p.text for p in all_positions],
        batch_size=32,
        normalize_embeddings=True,   # 必须 normalize，和 search() 里 query 的口径对上
    ).tolist()
    build_index(all_positions, vectors)

    col = _get_collection(DB_PATH, COLLECTION)
    print(f"【建库完成】块数 {len(all_positions)} | 库内条数 {col.count()}")
    assert col.count() == len(all_positions), "块数和库内条数不一致，建库不完整"


def retrieve_context(query: str, model, tokenizer):
    """返回 (safe_context, chunks)。查不到相关内容时返回 ("", [])。"""
    # 先看 top1 距离：太远说明库里没有，硬塞给模型只会让它编
    raw = search(query, model, top_k=1)
    top1 = raw[0][0] if raw else 2.0

    if top1 > DIST_REJECT:
        print(f"【检索】top1 {top1:.4f} > 拒答线 {DIST_REJECT}，按「知识库中没有」处理")
        return "", []

    chunks = search_as_chunks(query, model, top_k=TOP_K)

    if top1 > DIST_STRONG:
        print(f"【检索】top1 {top1:.4f} 在灰色区（{DIST_STRONG}~{DIST_REJECT}），答案可能不完整")
    else:
        print(f"【检索】top1 {top1:.4f}，强相关")

    token_limit = calc_available_chunk_quota(
        MODEL_MAX_WINDOW, SYSTEM_PROMPT, query, tokenizer, RESERVE_OUTPUT_TOKENS
    )
    # token_limit 可能 <= 0，这时 safe_context 是空串，下游照常走
    safe_context = clip_context_by_max_token(
        [c.pos.text for c in chunks], token_limit, tokenizer
    )
    return safe_context, chunks


def ask_llm(query: str, safe_context: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，检查 .env")
    prompt = build_rag_prompt(SYSTEM_PROMPT, safe_context, query)
    return llm_chat(api_key, MODEL, prompt, BASE_URL)


def print_sources(chunks) -> None:
    print("\n【出处】")
    if not chunks:
        print("  （无命中）")
        return
    for i, c in enumerate(chunks, 1):
        p = c.pos
        print(f"  [{i}] {Path(p.source).name}  第 {p.start}-{p.end} 字")
        if p.header:
            print(f"      小节：{p.header}")


def interactive_loop() -> None:
    """一次启动连续提问，模型只加载这一次。"""
    print(f"【模型】加载 {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    ensure_index(model)
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)

    print("输入问题回车；exit 退出。")
    while True:
        query = input("问题> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("再见。")
            break
        if not query:
            continue
        safe_context, chunks = retrieve_context(query, model, tokenizer)
        print(f"【答案】\n{ask_llm(query, safe_context)}")
        print_sources(chunks)


def main() -> None:
    if len(sys.argv) < 2:
        interactive_loop()
        return

    query = sys.argv[1].strip()
    if not query:
        print("问题不能为空")
        return

    print(f"【模型】加载 {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    ensure_index(model)

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    safe_context, chunks = retrieve_context(query, model, tokenizer)

    print(f"\n【问题】{query}")
    print(f"【检索】命中 {len(chunks)} 块，上下文 {len(tokenizer.encode(safe_context))} tokens")

    print(f"\n【答案】\n{ask_llm(query, safe_context)}")
    print_sources(chunks)


if __name__ == "__main__":
    main()
