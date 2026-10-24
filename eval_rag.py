import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from notes_loader import load_docs
from basic_rag_pipeline import build_corpus,SentenceTransformer,ChunkItem,retrieve
from pathlib import Path

docs,errors,bom=load_docs()
print(f"篇数={len(docs)},错误={len(errors)} BOM={bom}")

positions, report, failures = build_corpus(docs, 150, 30)
print(f"块数={len(positions)} 失败篇数={len(failures)}")

model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
vectors = model.encode([p.text for p in positions],batch_size=32).tolist()
print(f"向量数={len(vectors)}")

store = [ChunkItem(pos=p, vector=v) for p, v in zip(positions, vectors)]
K=5
result = retrieve("什么是RAG？",model,store,top_k=K)
for i,it in enumerate(result,1):
    print(f"TOP{i} {it.pos.source} [{it.pos.start},{it.pos.end}]")

EVAL_SET = [
    ("什么是RAG？", ["RAG六步链路·全局地图.md", "RAG系统课·第1讲.md", "Agent术语中英对照表.md"]),
    ("为什么说切分是RAG的命门？", ["RAG系统课·第1讲.md"]),
    ("为什么余弦相似度必须除以范数？", ["RAG后四环·向量化·检索·组装·生成.md"]),
    ("为什么overlap不能设得很大？", ["RAG系统课·第1讲.md"]),
    ("分块器有几道校验？分别是什么？", ["分块位置与可溯源.md", "分块与往返校验的实现.md", "项目README.md"]),
    ("语料加载层有几道哨兵？分别防什么？", ["元数据过滤与哨兵.md", "上下文检索·让块带着出处进向量.md"]),
    ("prev[-0:]为什么会出错？", ["RAG六步链路·全局地图.md", "RAG系统课·第1讲.md"]),
    ("删掉Observation那条tool消息，为什么第一反应是API报错？", ["tool消息的契约.md", "ReAct循环.md"]),        ("done字段为什么要用not？它怎么控制循环？", ["ReAct循环.md"]),
    ("checkpointer存的是最新一张快照还是全部？", ["checkpointer的两层存储.md", "checkpointer记忆持久化.md"]),
]

hit = 0
for q, gold in EVAL_SET:
    res = retrieve(q, model, store, top_k=K)
    got = [Path(it.pos.source).name for it in res]
    ok = any(g in got for g in gold)
    hit += 1 if ok else 0
    print(("PASS" if ok else "FAIL"), q)
    print("   期望:", gold)
    print("   实得:", got)

print(f"\nRecall@{K} = {hit}/{len(EVAL_SET)} = {hit / len(EVAL_SET) * 100:.0f}%")