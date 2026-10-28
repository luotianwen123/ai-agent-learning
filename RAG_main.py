# -*- coding: utf-8 -*-
"""RAG_main.py —— 一条命令问一个问题

用法：
    python RAG_main.py "什么是RAG？"

自带的冷启动行为：
    库不存在或为空 → 自动建库（约 60 秒）→ 继续回答
    库正常         → 直接检索回答

════════════════════════════════════════════════════════════════
【你要写的只有 4 个函数体】——都已标成 TODO，其余代码别改
────────────────────────────────────────────────────────────────
  TODO-A  ensure_index()      库不存在就现场建（冷启动）
  TODO-B  retrieve_context()  检索 + 取正文 + 按 token 预算裁剪
  TODO-C  ask_llm()           拼 prompt + 调模型
  TODO-D  print_sources()     把出处打成人能读的样子

════════════════════════════════════════════════════════════════
【验收清单】写完自己跑，5 条全过才算完
────────────────────────────────────────────────────────────────
  [ ] 1. python RAG_main.py                → 打印用法，不报错
  [ ] 2. python RAG_main.py "什么是RAG？"   → 出答案 + 5 条出处
  [ ] 3. 删掉/改名库目录再跑             → 自动重建并继续（**冷启动核心判据**）
  [ ] 4. 答案里有你笔记的内容
  [ ] 5. 出处文件名在 ai-agent-notes 里真实存在
"""

# ══════════════════════════════════════════════════════════════
# 区块 0 · 环境变量（必须在所有第三方 import 之前，顺序别动）
# ══════════════════════════════════════════════════════════════
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()   # ⚠️ 必须在 basic_rag_pipeline 之前：
                # 它在【模块级】就执行 API_KEY = os.getenv("OPENAI_API_KEY")，
                # 晚一步 load_dotenv，API_KEY 就永远是 None（报 401）。

# ══════════════════════════════════════════════════════════════
# 区块 1 · 标准库
# ══════════════════════════════════════════════════════════════
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 区块 2 · 第三方
# ══════════════════════════════════════════════════════════════
import tiktoken
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════
# 区块 3 · 本项目
# ══════════════════════════════════════════════════════════════
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
    search_as_chunks,
)

# ══════════════════════════════════════════════════════════════
# 区块 4 · 常量（照抄，别改数值）
# ══════════════════════════════════════════════════════════════
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # 与 chroma_build.py 完全一致
TOP_K = 5
MODEL_MAX_WINDOW = 8192                  # 保守值，别按 64k 算
RESERVE_OUTPUT_TOKENS = 512
TOKENIZER_NAME = "cl100k_base"

SYSTEM_PROMPT = (
    "你是知识库问答助手，只依据【参考文档】回答。"
    "文档里没有依据时，直接回答「知识库中没有相关内容」，不要编造。"
)


# ══════════════════════════════════════════════════════════════
# TODO-A · 冷启动：保证库存在且非空
# ══════════════════════════════════════════════════════════════
def ensure_index(model) -> None:
    """库不存在或为空 → 现场建库；否则直接返回。

    判据（三条都要满足才算"库可用"）：
      1. DB_PATH 目录存在
      2. collection 里 count() > 0
      3. 能查到东西（可选：建完立刻 search 一次自检）

    步骤：
      1) 判断：col = _get_collection(DB_PATH, COLLECTION)
                if DB_PATH.exists() and col.count() > 0: return
      2) print("首次运行，正在建库（约 60 秒）……")   ← 不打印会被当成卡死
      3) docs, errors, bom = load_docs()
      4) all_positions, report, failures = build_corpus(docs, MAX_CHUNK_SIZE, OVERLAP)
      5) assert not failures, "有 N 篇没通过逐篇校验，拒绝建残缺库"
      6) chunks = [p.text for p in all_positions]
      7) vectors = model.encode(chunks, batch_size=32,
                                normalize_embeddings=True).tolist()   ← 一定 normalize
      8) build_index(all_positions, vectors)
      9) print 对账：len(all_positions) 与 col.count() 是否一致
    """
    # ① 先看目录在不在。目录不存在时不能直接 _get_collection ——
    #    PersistentClient 会现场造一个空目录，get_collection 立刻抛 NotFoundError。
    if DB_PATH.exists():
        try:
            col = _get_collection(DB_PATH, COLLECTION)
            if col.count() > 0:
                return
        except Exception:
            # 目录在但 collection 不存在 / 索引坏了 → 当"不可用"，走重建
            pass

    # ② 建库前先说话，否则 60 秒没输出会被当成卡死
    print("首次运行，正在建库（约 60 秒）……")

    # ③ 读语料
    docs, errors, bom = load_docs()
    if errors:
        print(f"  ⚠️ 有 {len(errors)} 篇没读进来（不影响建库，但语料少了）")

    # ④ 分块 + 逐篇四道校验
    all_positions, report, failures = build_corpus(docs, MAX_CHUNK_SIZE, OVERLAP)

    # ⑤ 有一篇失败就拒绝建库：残缺库比没库更难查
    assert not failures, f"有 {len(failures)} 篇没通过逐篇校验，拒绝建残缺库"

    # ⑥⑦ 编码：必须 normalize_embeddings=True，
    #      和 chroma_store.search() 里 query 的归一化保持一致，否则余弦距离没意义
    chunks = [p.text for p in all_positions]
    vectors = model.encode(
        chunks, batch_size=32, normalize_embeddings=True
    ).tolist()

    # ⑧ 写库（build_index 内部会先删目录再整批 add）
    build_index(all_positions, vectors)

    # ⑨ 对账：块数 ≠ 库里条数就说明写漏了
    col = _get_collection(DB_PATH, COLLECTION)
    print(f"【建库完成】块数 {len(all_positions)} | 库内条数 {col.count()}")
    assert col.count() == len(all_positions), "块数与库内条数不一致，建库不完整"


# ══════════════════════════════════════════════════════════════
# TODO-B · 检索 + 裁剪
# ══════════════════════════════════════════════════════════════
def retrieve_context(query: str, model, tokenizer):
    """返回 (safe_context, chunks)

    safe_context : str            —— 已按 token 预算裁剪好的上下文
    chunks       : list[ChunkItem] —— 原始命中块（用于打印出处）

    步骤：
      1) chunks = search_as_chunks(query, model, top_k=TOP_K)
      2) texts = [c.pos.text for c in chunks]
      3) token_limit = calc_available_chunk_quota(
                        MODEL_MAX_WINDOW, SYSTEM_PROMPT, query,
                        tokenizer, RESERVE_OUTPUT_TOKENS)
      4) safe_context = clip_context_by_max_token(texts, token_limit, tokenizer)
      5) return safe_context, chunks

    注意：token_limit 可能算成 0 或负数 → 这时 safe_context 会是空串，
          要能正常走下去（模型会回答"知识库中没有相关内容"），不许崩。
    """
    # ① 检索（内部已经 normalize 过 query 向量）
    chunks = search_as_chunks(query, model, top_k=TOP_K)

    # ② 取正文
    texts = [c.pos.text for c in chunks]

    # ③ 算还剩多少 token 能给参考文档
    token_limit = calc_available_chunk_quota(
        MODEL_MAX_WINDOW, SYSTEM_PROMPT, query,
        tokenizer, RESERVE_OUTPUT_TOKENS,
    )

    # ④ 按预算裁剪；token_limit <= 0 时返回空串，下游照样能走
    safe_context = clip_context_by_max_token(texts, token_limit, tokenizer)

    return safe_context, chunks


# ══════════════════════════════════════════════════════════════
# TODO-C · 拼 prompt + 调模型
# ══════════════════════════════════════════════════════════════
def ask_llm(query: str, safe_context: str) -> str:
    """返回模型答案字符串。

    步骤：
      1) api_key = os.environ.get("OPENAI_API_KEY")
         if not api_key: raise RuntimeError("缺少 OPENAI_API_KEY，检查 .env")
      2) prompt = build_rag_prompt(SYSTEM_PROMPT, safe_context, query)
      3) reply = llm_chat(api_key, MODEL, prompt, BASE_URL)
      4) return reply

    参数顺序别记错：llm_chat(bearer_key, model_name, prompt_text, base_url)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，检查 .env")

    prompt = build_rag_prompt(SYSTEM_PROMPT, safe_context, query)
    reply = llm_chat(api_key, MODEL, prompt, BASE_URL)
    return reply


# ══════════════════════════════════════════════════════════════
# TODO-D · 打印出处
# ══════════════════════════════════════════════════════════════
def print_sources(chunks) -> None:
    """把出处打成人能读的样子。

    期望输出（名字 + 字符区间，不要整条路径）：
      【出处】
        [1] 分块位置与可溯源.md  第 1234-1390 字
            小节：三、四道校验
        [2] ...

    可用字段：c.pos.source / c.pos.start / c.pos.end / c.pos.header
    """
    print("\n【出处】")
    if not chunks:
        print("  （无命中）")
        return

    for i, c in enumerate(chunks, 1):
        p = c.pos
        print(f"  [{i}] {Path(p.source).name}  第 {p.start}-{p.end} 字")
        if p.header:
            print(f"      小节：{p.header}")


# ══════════════════════════════════════════════════════════════
# 主流程（已写好，别改）
# ══════════════════════════════════════════════════════════════
def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python RAG_main.py "你的问题"')
        return
    query = sys.argv[1].strip()
    if not query:
        print("问题不能为空")
        return

    print(f"【模型】加载 {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)

    ensure_index(model)                      # TODO-A

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    safe_context, chunks = retrieve_context(query, model, tokenizer)   # TODO-B

    print(f"\n【问题】{query}")
    print(f"【检索】命中 {len(chunks)} 块，上下文 {len(tokenizer.encode(safe_context))} tokens")

    answer = ask_llm(query, safe_context)    # TODO-C
    print(f"\n【答案】\n{answer}")

    print_sources(chunks)                    # TODO-D


if __name__ == "__main__":
    main()
