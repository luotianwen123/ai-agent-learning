r"""
chroma_demo.py —— Chroma 最小可用例子
目的：看明白"向量存在磁盘上"和"存在内存 list 里"差在哪
运行：在项目目录下，& "...\.venv\Scripts\python.exe" chroma_demo.py
"""

import os

# 关键！必须写在 import sentence_transformers 之前
# 不设的话它会去 huggingface.co 检查模型更新，国内直连超时 → WinError 10060 重试 5 次
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

client = chromadb.PersistentClient(path="./chroma_demo_db")

collection = client.get_or_create_collection(name="demo")

docs = [ "RAG 是检索增强生成，把外部知识塞进提示词里再让模型回答",
    "LangGraph 用状态图编排 Agent 的节点流转",
    "今天中午投了两份简历给郑州的公司",]

vectors = model.encode(docs,normalize_embeddings=True)
#collection.add(ids=["1", "2", "3"], documents=docs, embeddings=vectors.tolist())

q=model.encode(["怎么把知识存进去再查出来"], normalize_embeddings=True)
result = collection.query(query_embeddings=q.tolist(),n_results=2)

for i, (doc, dist) in enumerate(zip(result["documents"][0], result["distances"][0])):
    print(f"[{i+1}] 距离={dist:.4f}  {doc}")