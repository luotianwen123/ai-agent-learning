r"""
chroma_demo.py —— Chroma 最小可用例子
目的：看明白"向量存在磁盘上"和"存在内存 list 里"差在哪
运行：& "...\PyCharmMiscProject\.venv\Scripts\python.exe" chroma_demo.py
      （在哪个目录跑都行 —— 路径由下面 DB_DIR 决定，不依赖当前工作目录）
"""

import os
from pathlib import Path

# 关键！必须写在 import sentence_transformers 之前
# 不设的话它会去 huggingface.co 检查模型更新，国内直连超时 → WinError 10060 重试 5 次
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

# 🔴 放纯英文路径（家目录），不放项目目录
# 项目目录叫 "ai agent的初次尝试" —— 有中文、有空格。
# Chroma 1.x 的 rust 后端处理非 ASCII 路径会报 Error loading hnsw index
# （现象：同一进程刚写完能查，换个进程就读不出来）。详见 chroma_store.py 顶部说明。
DB_DIR = Path(os.environ.get("CHROMA_DEMO_PATH", str(Path.home() / "chroma_demo_db")))

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

client = chromadb.PersistentClient(path=str(DB_DIR))

# 幂等：每次跑先清掉同名 collection
# 不清的话，第二次跑会报 "IDs ['1','2','3'] already exist"。
# 这和 build_index 里"整个目录删掉重建"是同一个道理：demo 要能重复跑。
try:
    client.delete_collection(name="demo")
except Exception:
    pass  # 第一次跑时 collection 还不存在，正常
collection = client.create_collection(name="demo")

docs = [
    "RAG 是检索增强生成，把外部知识塞进提示词里再让模型回答",
    "LangGraph 用状态图编排 Agent 的节点流转",
    "今天中午投了两份简历给郑州的公司",
]

vectors = model.encode(docs, normalize_embeddings=True)
collection.add(ids=["1", "2", "3"], documents=docs, embeddings=vectors.tolist())

print(f"【写入】库路径 = {DB_DIR}")
print(f"【对账】库内条数 = {collection.count()}  期望 3 → "
      f"{'一致 ✓' if collection.count() == 3 else '不一致 ✗'}")

q = model.encode(["怎么把知识存进去再查出来"], normalize_embeddings=True)
result = collection.query(query_embeddings=q.tolist(), n_results=2)

print("\n【检索】query = 怎么把知识存进去再查出来")
for i, (doc, dist) in enumerate(zip(result["documents"][0], result["distances"][0]), 1):
    print(f"  [{i}] 相似度={1 - dist:.4f}  {doc}")
