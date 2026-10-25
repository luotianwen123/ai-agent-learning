# 【强制第1行】加载环境变量，必须在所有第三方import之前
from dotenv import load_dotenv
load_dotenv()

import os

# 必须在 sentence_transformers 之前，否则它会去 huggingface.co 查更新 → WinError 10060
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time

_T0 = time.perf_counter()  # 量启动耗时，从这一行开始算

from fastapi import FastAPI
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer
from notes_loader import load_docs
from basic_rag_pipeline import (
    row_num,
    calc_available_chunk_quota,
    clip_context_by_max_token,
    build_rag_prompt,
    llm_chat,
    API_KEY,
    BASE_URL,
    MODEL,
)
from chroma_store import search_as_chunks, DB_PATH, COLLECTION, _get_collection
import uvicorn

app = FastAPI()

# ========== 全局一次性初始化 ==========
docs, _, _ = load_docs()
path2text = {d["path"]: d["text"] for d in docs}
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
tokenizer = tiktoken.get_encoding("cl100k_base")

# ========== 向量库：从磁盘上的 Chroma 读，不再每次重算 ==========
# 改动前：启动时要 build_corpus + encode 3193 个块，几十秒
# 改动后：只打开磁盘上的索引，毫秒级 —— 向量在建库时算过一次就够了
_col = _get_collection(DB_PATH, COLLECTION)
_INDEX_SIZE = _col.count()
if _INDEX_SIZE == 0:
    raise RuntimeError("向量库是空的，先跑一次 chroma_build.py")
print(f"【向量库】已加载 {_INDEX_SIZE} 条  ← {DB_PATH}")
print(f"【启动耗时】{time.perf_counter() - _T0:.2f} 秒")


@app.post("/ask")
def ask(q: str):
    # 检索：改成从磁盘上的 Chroma 查（返回的仍是 list[ChunkItem]，下游不用改）
    hit_chunks = search_as_chunks(q, embedding_model, top_k=5)

    # 出处信息
    sources = [
        {
            "file_name": Path(it.pos.source).name,
            "row_num": row_num(path2text[it.pos.source], it.pos.start),
            "header": it.pos.header
        }
        for it in hit_chunks
    ]

    # 提取块文本
    texts = [it.pos.text for it in hit_chunks]

    # token配额计算 + 上下文裁剪
    available = calc_available_chunk_quota(
        model_max_window=8192,
        system_prompt="你是RAG知识库助手，基于给定文档回答问题",
        user_query=q,
        tokenizer=tokenizer,
        reserve_output_token=1024
    )
    safe_context = clip_context_by_max_token(texts, available, tokenizer=tokenizer)

    # 组装提示词
    prompt = build_rag_prompt(
        "你是RAG知识库助手，基于给定文档回答问题",
        safe_context,
        q
    )

    # 调用大模型
    answer = llm_chat(
        bearer_key=API_KEY,
        model_name=MODEL,
        prompt_text=prompt,
        base_url=BASE_URL
    )
    return {"answer": answer, "sources": sources}


# ========== 启动入口 ==========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
