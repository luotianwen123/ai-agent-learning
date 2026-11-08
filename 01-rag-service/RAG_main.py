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
    rows_to_chunks,
)

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# ---- 检索两段式：粗排 RECALL_K 条 → 精排（rerank）→ 取 TOP_K 条 ----
# 为什么要粗排：向量检索只能给「像不像」，给不了「答不答得上」。
#   一块「很像但只答一半」的文本会挤掉真正补全答案的那块，所以先多捞。
# 参照 anything-llm（67.8k★）server/utils/vectorDbProviders/lance/index.js
#   第 129-132 行：searchLimit = clamp(总块数 × 0.1, 10, 50)
#   它的库普遍很小，所以取 50 上限；本项目 2978 块也顶到 50。
#   本项目实测定档 50：2026-11-02 15:15 在 Ryzen 7 7735H 上实测
#   5 对 0.53s / 20 对 1.28s / 50 对 3.03s（每对约 61ms）⇒ 50 条完全跑得动。
#   这只是打分耗时，不含向量检索和 LLM 生成。
TOP_K = 5              # 精排后真正喂给模型的条数
RECALL_K = 50          # 粗排候选数（只给 reranker 看，不进上下文）

# 重排开关（跟 OVERLAP 一样，一个开关决定开不开）—— 默认关。
# 2026-11-02 实测（10 题评测集）：MRR 0.633 → 0.537，recall@5 1.0 → 0.9，2 好 4 坏。
#   根因：重排把多个名额给了同一篇笔记（体裁像答案的「面试问答」），
#   挤掉了散在其他文件里的答案块 —— 碎片化笔记语料上，重排会压平结果多样性。
# 想开：终端先 $env:RAG_RERANK="1" 再跑，不用改代码。
RERANK_ENABLED = os.environ.get("RAG_RERANK", "0") == "1"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

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


_RERANKER = None


def get_reranker():
    """懒加载重排模型：第一次真正用到才加载，不拖慢启动。"""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        print(f"【模型】加载重排模型 {RERANK_MODEL} …（首次会下载，约 2.3G）")
        _RERANKER = CrossEncoder(RERANK_MODEL)
    return _RERANKER


def rerank_chunks(query, chunks, top_n):
    """用交叉编码器给「问题 + 块」逐对打分，返回分数最高的 top_n 条。

    和向量检索的区别：
      向量检索 = 双塔（bi-encoder，两座塔各编各的），问题和块分开编码，谁也没见过谁；
      重排     = 交叉（cross-encoder，把两者拼成一条一起过模型）。
    所以它能判「这块答不答得上」，而不只是「这块像不像」。
    """
    if len(chunks) <= 1:
        return chunks[:top_n]

    reranker = get_reranker()
    pairs = [(query, c.pos.text) for c in chunks]
    scores = reranker.predict(pairs)

    # 按分数从高到低排；sorted 是稳定排序，分数相同则保持原顺序
    order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
    return [chunks[i] for i in order[:top_n]]


def retrieve_context(query: str, model, tokenizer):
    """返回 (safe_context, chunks)。查不到相关内容时返回 ("", [])。

    检索两段式（开关 RAG_RERANK 控制，默认关，用环境变量打开）：
        ① 粗排：一次查 recall_k 条（开关打开 = RECALL_K 50 条，关闭 = TOP_K 5 条）
        ② 闸门：top1 距离 > DIST_REJECT ⇒ 判定库里没有，直接返回空，模型不参与
        ③ 过滤：逐条丢掉距离 > DIST_REJECT 的块，不让它占重排名额
        ④ 精排（仅开关打开时）：交叉编码器给「问题 + 块」逐对打分，取前 TOP_K 条

    为什么默认关：2026-11-02 实测同一份 10 题评测集，
    开重排 MRR 0.633 → 0.537、recall@5 1.0 → 0.9。
    根因是重排压平了结果多样性 —— 它偏好「体裁像答案」的块（问答体笔记），
    把散在其它文件里的答案块挤走。碎片化笔记语料上，这等于删信息。
    详见 README 变更卡 17。
    """
    # 一次查够：闸门和粗排共用同一批结果（原来是查两次库）
    #   重排关闭时只取 TOP_K 条 —— 行为与加重排之前完全一致
    recall_k = RECALL_K if RERANK_ENABLED else TOP_K
    raw = search(query, model, top_k=recall_k)
    if not raw:
        return "", []

    top1 = raw[0][0]

    # 闸门：top1 太远说明库里没有，硬塞给模型只会让它编
    if top1 > DIST_REJECT:
        print(f"【检索】top1 {top1:.4f} > 拒答线 {DIST_REJECT}，按「知识库中没有」处理")
        return "", []

    # 逐条过滤：超过拒答线的块先丢掉，别让它占掉重排的名额
    #   参照 anything-llm 第 139 行注释：
    #   "Drop pinned and below-threshold chunks first so they cannot take the reranker's topK slots."
    kept = [row for row in raw if row[0] <= DIST_REJECT]
    dropped = len(raw) - len(kept)
    if not kept:
        print(f"【检索】{len(raw)} 条候选全部超过拒答线，按「知识库中没有」处理")
        return "", []

    pool = rows_to_chunks(kept)

    # 精排：粗排 RECALL_K 条 → 重排 → 只留 TOP_K 条
    if RERANK_ENABLED and len(pool) > TOP_K:
        chunks = rerank_chunks(query, pool, TOP_K)
        print(f"【检索】粗排 {len(raw)} 条 → 过滤掉 {dropped} 条 → 重排 → 留 {len(chunks)} 条")
    else:
        chunks = pool[:TOP_K]
        print(f"【检索】粗排 {len(raw)} 条 → 过滤掉 {dropped} 条 → 未重排，直接留 {len(chunks)} 条")

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
