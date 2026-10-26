"""
chroma_store.py —— 把向量存到磁盘上的库（Chroma），提供"建库"和"检索"两个函数
目的：项目现在的向量存在内存 list 里，脚本一退就没了；换成磁盘上的库之后，算一次永久复用
      "向量数据库" 也是三份郑州 JD 里 3/3 点名的东西

设计要点：
  1. hnsw:space="cosine" —— 显式指定用余弦距离，避免默认 L2 带来的"越小越像/越大越像"混淆
     cosine 空间下： distance = 1 - 余弦相似度 ，范围 [0, 2]
  2. metadata 只能是 str / int / float / bool，不能是 None，所以 header 用 "" 兜底
"""

import os
from pathlib import Path

# 必须写在 import sentence_transformers 之前，否则它会去 huggingface.co 查更新 → WinError 10060
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb

# 🔴 用 __file__ 定位，不写 "./xxx"
# 写 "./" 的意思是"我敲命令时所在的目录"，不是"这个脚本所在的目录"
#
# 🔴🔴 而且故意放在纯英文路径下，不放项目目录
# 项目目录叫 "ai agent的初次尝试" —— 有中文、有空格。
# Chroma 1.x 的 rust 后端处理非 ASCII 路径时会出问题：
#   报 Error loading hnsw index / backfill request to compactor，
#   现象就是"同一个进程刚写完能查，换个进程就读不出来"。
# 排查顺序里，这种"环境里的小变量"要最先排除，不要一开始就怀疑库有 bug。
DB_PATH = Path(os.environ.get("CHROMA_DB_PATH", str(Path.home() / "chroma_notes_db")))
COLLECTION = "notes"

# 分块参数：建库（chroma_build.py）与启动对账（FAST_API.py）必须用【同一组数】，
# 否则「块数 vs 库里条数」永远对不上，会天天误报。
# 放这里 = 单一真源；别在两个文件里各写一份（改一处忘一处就是这类 bug 的来源）。
MAX_CHUNK_SIZE = 150
OVERLAP = 30

# 复用一个 client，别每次查询都新开一遍
# 每次 PersistentClient() 都要重新打开 sqlite + 加载索引，单次 50–200ms，
# 放进 FastAPI 就是每个请求白付一次。缓存之后只付第一次。
_COLLECTION_CACHE: dict = {}


def _get_collection(path, collection_name):
    key = (str(path), collection_name)
    if key not in _COLLECTION_CACHE:
        client = chromadb.PersistentClient(path=str(path))
        _COLLECTION_CACHE[key] = client.get_collection(name=collection_name)
    return _COLLECTION_CACHE[key]


def build_index(all_positions, vectors, path=DB_PATH, collection_name=COLLECTION):
    """把块和向量写进磁盘上的库。

    两个关键改动（都是被 bug 逼出来的）：
      1. 整个目录删掉重建，不用 delete_collection()
         —— 只删表会在磁盘上留下残留 segment 文件，下次加载 index 时会把它读出来并炸掉
      2. 一次性 add，不分批
         —— 分批写会生成多个 segment，跨进程加载 hnsw 索引时只要有一个段读不出来就整体失败，
            报错长这样：Error loading hnsw index
         3193 × 512 维 ≈ 6MB，一次写完毫无压力，分批纯属自找麻烦
    """
    import shutil

    db_path = Path(str(path))
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
        except PermissionError as e:
            # Windows 特有：库文件被占用时 rmtree 只会抛 WinError 32，看不出是谁占的
            # 最常见占用者 = 正在跑的 FAST_API.py / uvicorn（它 PersistentClient 一直开着 sqlite）
            raise RuntimeError(
                f"删不掉旧库：{db_path}\n"
                f"原因：有进程正占用库文件（最常见的是还在跑的 FAST_API.py / uvicorn）。\n"
                f"修法：先停掉服务（PyCharm 里点红色方块，或关掉那个终端窗口），再重跑本脚本。\n"
                f"原始错误：{e}"
            ) from e

    _COLLECTION_CACHE.clear()  # 库被删了，缓存里的句柄也失效了

    client = chromadb.PersistentClient(path=str(db_path))
    col = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # 距离 = 1 - 相似度
    )

    ids = [f"{p.source}#{p.index}" for p in all_positions]
    documents = [p.text for p in all_positions]
    metadatas = [
        {
            "source": str(p.source),
            "header": str(p.header) if p.header else "",
            "start": int(p.start),
            "end": int(p.end),
            "truncated": bool(p.truncated),
            "actual_overlap": int(p.actual_overlap),
            "index": int(p.index),
        }
        for p in all_positions
    ]

    col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)

    return col


def search(query, model, top_k=5, path=DB_PATH, collection_name=COLLECTION):
    """从磁盘上的库检索。返回 [(距离, 文本, 元数据), ...]，按距离升序（越小的越相关）。"""
    col = _get_collection(path, collection_name)

    q = model.encode([query], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=q, n_results=top_k)

    return list(zip(r["distances"][0], r["documents"][0], r["metadatas"][0]))


def search_as_chunks(query, model, top_k=5, path=DB_PATH, collection_name=COLLECTION):
    """检索后把结果包装回 ChunkItem，让下游代码一行都不用改。

    为什么要这一层：
      原来的 retrieve() 返回 list[ChunkItem]，FAST_API 里到处用 it.pos.xxx 取值。
      换成 Chroma 后返回的是 (距离, 文本, 元数据)。如果直接改下游，
      就要在 /ask 里改四五处取值逻辑，出错面更大。
      包一层适配器，做到"接口不变、实现替换" —— 这是改造老项目的标准手法。
    """
    # 函数内 import，避免 chroma_store 在顶层依赖 basic_rag_pipeline 造成循环引用
    from basic_rag_pipeline import ChunkItem, ChunkPosition

    hits = search(query, model, top_k=top_k, path=path, collection_name=collection_name)

    return [
        ChunkItem(
            pos=ChunkPosition(
                text=text,
                start=int(m["start"]),
                end=int(m["end"]),
                actual_overlap=int(m.get("actual_overlap", -1)),
                truncated=bool(m["truncated"]),
                header=m["header"],
                source=m["source"],
                index=int(m.get("index", -1)),
            ),
            vector=[],  # 检索结果不需要再带向量，省内存
        )
        for dist, text, m in hits
    ]
