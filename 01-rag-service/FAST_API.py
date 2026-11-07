# 【强制第1行】加载环境变量，必须在所有第三方import之前
from dotenv import load_dotenv
load_dotenv()

import os

# 必须在 sentence_transformers 之前，否则它会去 huggingface.co 查更新 → WinError 10060
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time

_T0 = time.perf_counter()  # 量启动耗时，从这一行开始算
_BOOT_AT = time.time()     # 量"服务已经活了多久"（/health 用），和上面那个不是同一个东西

from fastapi import FastAPI, Response
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
from chroma_store import (
    search_as_chunks, DB_PATH, COLLECTION, _get_collection,
    MAX_CHUNK_SIZE, OVERLAP,   # 分块参数（单一真源在 chroma_store）
)
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
# ========== 启动自检：当前语料的分块块数 vs 库里条数 ==========
# 为什么要有这一步：
#   库里存的是「上一次建库那一刻」的块。如果之后语料改了（加/删笔记）、或分块参数改了，
#   而没人重跑 chroma_build.py —— 服务照样启动、照样回答，但答的是【旧库】，
#   而且【完全不报错】。这就是静默错：看起来一切正常，实际答的是过期数据。
#   所以启动时用同一组参数重新分块一次（只分块、不算向量，秒级），拿块数跟库里对账。
def _corpus_chunk_count():
    """返回 (块数, 失败篇数)。只分块不 encode，所以很快。"""
    import io
    from contextlib import redirect_stdout
    from basic_rag_pipeline import build_corpus

    buf = io.StringIO()
    # build_corpus 内部每篇会打印调试行；这里吞掉，否则启动日志里会多几百行噪音
    with redirect_stdout(buf):
        positions, _report, failures = build_corpus(docs, MAX_CHUNK_SIZE, OVERLAP)
    return len(positions), len(failures)


_A, _FAILS = _corpus_chunk_count()
if _FAILS:
    print(f"【启动自检 ⚠️】有 {_FAILS} 篇没通过分块校验 —— 下面的块数对账结果不可信")
if _A != _INDEX_SIZE:
    print("=" * 66)
    print(f"【启动自检 ❌】库与当前语料不一致：语料分块 {_A} 块，库里 {_INDEX_SIZE} 条")
    print(f"   差额 {_A - _INDEX_SIZE:+d} 条")
    print("   原因：语料改过 / 分块参数改过，但没重建库（服务会继续用旧库回答）")
    print("   修法：先跑一次  python chroma_build.py  ，再重启本服务")
    print("=" * 66)
else:
    print(f"【启动自检 ✅】分块 {_A} 块 = 库里 {_INDEX_SIZE} 条 —— 库与语料一致")

_BOOT_SEC = round(time.perf_counter() - _T0, 2)
print(f"【启动耗时】{_BOOT_SEC} 秒")


# ========== 运维三件套：让服务"出问题能查" ==========
# 加这三样的原因：
#   改动前 —— 服务答错了，你只能去翻启动时那一屏日志，翻不到就只剩"感觉不对"。
#   改动后 —— 任何人在任何机器上 curl 一下 /health，就能看见库对不对得上。
#   这也是「工程级」和「能跑」的分水岭：接手的人不用读你代码就知道服务健不健康。

@app.get("/")
def root():
    """不裸 404：告诉调用方这服务是什么、有哪些接口。"""
    return {
        "service": "notes-rag",
        "desc": "把本地 Obsidian 笔记库做成可问答的 RAG 服务",
        "endpoints": {
            "GET /": "本页（接口清单）",
            "GET /health": "健康检查：库与语料是否对得上",
            "POST /ask?q=你的问题": "问答，返回 answer + sources",
        },
    }


@app.get("/favicon.ico")
def favicon():
    """浏览器每个页面都会自动请求这个。不处理就是一条 404 噪音，会把真错误刷走。"""
    return Response(status_code=204)


@app.get("/health")
def health():
    """探针接口。判据：status == "ok" 才算健康，degraded 表示库和语料对不上。

    为什么把「对账结果」也吐出来：
      只看 HTTP 200 是不够的 —— 库过期时服务照样 200、照样答，只是答的是旧数据。
      把 corpus.chunks 和 index.chunks 并排放出来，一眼就能看出差额。
    """
    consistent = _A == _INDEX_SIZE
    return {
        "status": "ok" if consistent else "degraded",
        "corpus": {
            "docs": len(docs),
            "chunks": _A,
            "chunk_failures": _FAILS,
        },
        "index": {
            "chunks": _INDEX_SIZE,
            "path": str(DB_PATH),
            "collection": COLLECTION,
        },
        "chunking": {
            "max_chunk_size": MAX_CHUNK_SIZE,
            "overlap": OVERLAP,
        },
        "boot_sec": _BOOT_SEC,
        "uptime_sec": round(time.time() - _BOOT_AT, 1),
        # 不一致时直接把修法写进来，省得下次还得回想
        "hint": "" if consistent else f"差额 {_A - _INDEX_SIZE:+d} 条：先跑 python chroma_build.py 再重启",
    }



@app.post("/ask")
def ask(q: str):
    _t = time.perf_counter()   # 给每个请求埋一条日志：线上慢的时候才知道慢在哪一段

    # 检索：改成从磁盘上的 Chroma 查（返回的仍是 list[ChunkItem]，下游不用改）
    hit_chunks = search_as_chunks(q, embedding_model, top_k=5)
    _t_retrieve = time.perf_counter()

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

    # 一行日志 = 一次请求的可追溯记录。
    # 格式固定、字段固定，别随手改 —— 以后要 grep "检索慢" 就是靠这个前缀。
    # 检索段(ms) 和 生成段(ms) 分开记：慢在检索还是慢在模型，一眼分得清。
    _now = time.perf_counter()
    print(
        f"【/ask】q={q[:30]!r} | 块 {len(hit_chunks)} | 上下文 {len(safe_context)} 段 | "
        f"检索 {( _t_retrieve - _t) * 1000:.0f}ms | 生成 {(_now - _t_retrieve) * 1000:.0f}ms | "
        f"共 {(_now - _t) * 1000:.0f}ms"
    )

    return {"answer": answer, "sources": sources}


# ========== 启动入口 ==========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
