"""
chroma_build.py —— 建库脚本：跑一次就行，之后不用再跑
目的：把当前 57 篇 / 约 3100+ 个块的真实向量写进磁盘上的 Chroma

判据：最后一行打印的条数 = build_corpus 算出的块数
      并且项目目录下出现 chroma_notes_db 文件夹，里面有 chroma.sqlite3

注意：第一次 encode 3100+ 块要几十秒到两分钟，CPU 跑的，进度条不动不是卡死
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

from notes_loader import load_docs
from basic_rag_pipeline import build_corpus
from chroma_store import build_index

MAX_CHUNK_SIZE = 150
OVERLAP = 30

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

docs, errors, bom_files = load_docs()
all_positions, report, failures = build_corpus(docs, MAX_CHUNK_SIZE, OVERLAP)
assert not failures, f"有 {len(failures)} 篇没通过逐篇校验"

chunks = [p.text for p in all_positions]
print(f"\n【编码】{len(chunks)} 个块，开始算向量……")
vectors = model.encode(chunks, batch_size=32, normalize_embeddings=True).tolist()

col = build_index(all_positions, vectors)

print(f"\n【建库完成】collection={col.name!r}  写入条数 = {col.count()}")
print(f"【对账】build_corpus 块数 = {len(all_positions)} | 库里条数 = {col.count()}"
      f" → {'一致 ✓' if col.count() == len(all_positions) else '不一致 ✗'}")

# ===== 建完立刻查一次 =====
# 为什么：col.count() 读的是元数据，能返回 3193 不代表 hnsw 索引真的落盘了。
# 上次就是 count 说 3193、下一次查询却报 "Error loading hnsw index"。
# 建完当场查一次，索引有问题当场暴露，不用等下一轮。
from pathlib import Path

from chroma_store import search

hits = search("什么是RAG？", model, top_k=3)
print("\n【即时验证】建库后立刻检索：")
for i, (dist, text, m) in enumerate(hits, 1):
    print(f"  {i}. 相似度={1 - dist:.4f}  {Path(m['source']).name}  start={m['start']}")

print("\n【结论】" + ("索引可用，检索正常 ✓" if hits else "索引可疑，查不出东西 ✗"))

