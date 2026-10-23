# 【强制第1行】加载环境变量，必须在所有第三方import之前
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer
from notes_loader import load_docs
from basic_rag_pipeline import (
    build_corpus,
    ChunkPosition,
    ChunkItem,  # ① 补充分块向量条目类型
    retrieve,
    row_num,
    calc_available_chunk_quota,
    clip_context_by_max_token,
    build_rag_prompt,
    llm_chat,
    API_KEY,
    BASE_URL,
    MODEL
)
import uvicorn

app = FastAPI()

# ========== 全局一次性初始化 ==========
docs, _, _ = load_docs()
path2text = {d["path"]: d["text"] for d in docs}
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
tokenizer = tiktoken.get_encoding("cl100k_base")
all_positions, report, failures = build_corpus(docs, 150, 30)
assert not failures, f"文档分块校验失败: {failures}"

# ========== ② 替换为标准向量库构建（批量编码+ChunkItem封装，四行结构） ==========
embeddings = embedding_model.encode([p.text for p in all_positions], normalize_embeddings=True)
vector_store = [
    ChunkItem(pos=p, vector=e)
    for p, e in zip(all_positions, embeddings)
]


@app.post("/ask")
def ask(q: str):
    # 检索：签名 retrieve(query, model, vector_store, top_k)
    hit_chunks = retrieve(q, embedding_model, vector_store, top_k=5)

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
