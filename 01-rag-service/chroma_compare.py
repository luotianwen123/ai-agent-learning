"""
chroma_compare.py —— 新旧检索对账：证明换成 Chroma 之后，检索结果没变
目的：这是"改了代码要证明没改坏"的落地。老实现（内存暴力算余弦）当标准答案，
      新实现（Chroma）去对答案。对不上就说明有 bug，不许往下走。

判据：两条路径给出的 top5（文件名 + 起始位置）完全一致
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

from sentence_transformers import SentenceTransformer

from notes_loader import load_docs
from basic_rag_pipeline import build_corpus, retrieve, ChunkItem
from chroma_store import search

MAX_CHUNK_SIZE = 150
OVERLAP = 30
TOP_K = 5
QUERY = "什么是RAG？"

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

docs, _, _ = load_docs()
all_positions, _, failures = build_corpus(docs, MAX_CHUNK_SIZE, OVERLAP)
assert not failures

chunks = [p.text for p in all_positions]
vectors = model.encode(chunks, batch_size=32, normalize_embeddings=True).tolist()
vector_store = [ChunkItem(pos=p, vector=v) for p, v in zip(all_positions, vectors)]

# ---- 老实现：内存里逐条算余弦 ----
old = retrieve(QUERY, model, vector_store, top_k=TOP_K)
old_keys = [(p.pos.source, p.pos.start) for p in old]

# ---- 新实现：Chroma ----
new = search(QUERY, model, top_k=TOP_K)
new_keys = [(m["source"], m["start"]) for _, _, m in new]

print(f"\nquery = {QUERY!r}   top_k = {TOP_K}")

print("\n【老实现 · 内存余弦】")
for i, p in enumerate(old, 1):
    print(f"  {i}. {Path(p.pos.source).name}  start={p.pos.start}  {p.pos.header}")

print("\n【新实现 · Chroma】")
for i, (dist, text, m) in enumerate(new, 1):
    print(f"  {i}. {Path(m['source']).name}  start={m['start']}  "
          f"相似度={1 - dist:.4f}  {m['header']}")

ok = old_keys == new_keys
print(f"\n【对账】{'完全一致 ✓' if ok else '不一致 ✗'}")
if not ok:
    for i, (a, b) in enumerate(zip(old_keys, new_keys), 1):
        if a != b:
            print(f"  第 {i} 位不同：老 = {a}   新 = {b}")
