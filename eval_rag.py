# -*- coding: utf-8 -*-
"""RAG 检索测评线（EVAL_SET → Chroma 检索 → 指标 → 存档对比）。

四层结构：
  ① 数据层 EVAL_SET     题 + gold（文件 + 答案里必含的逐字片段）
  ② 检索层 make_searcher 给 query 返回按相似度排好的块（换实现只改这里）
  ③ 指标层 _match_chunk / compute_metrics
  ④ 报告层 print_table / save_run / compare_last_run
"""
import json
import os
from datetime import datetime
from pathlib import Path

# 必须在 sentence_transformers 之前
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

from notes_loader import load_docs
from chroma_store import search_as_chunks, _get_collection, DB_PATH, COLLECTION

K_RETRIEVE = 10   # 先捞 10 条，才能算出排第几
K_REPORT = 5      # 对外报 Recall@5 / Hit@5
RUNS_DIR = Path(__file__).with_name("eval_runs")

# must_contain 必须是【从笔记里逐字复制】的片段；手打/总结出来的词必然 0 次命中。
# 同一题允许多个 (文件, 关键词) 组合，命中任一即算对。
EVAL_SET = [
    {"id": "Q1", "q": "RAG 和微调该怎么选？", "gold": [
        {"file": "RAG系统课·第1讲.md", "must_contain": "行为风格"},
    ]},
    {"id": "Q2", "q": "为什么说切分是RAG的命门？", "gold": [
        {"file": "RAG系统课·第1讲.md", "must_contain": "整篇塞进去"},
    ]},
    {"id": "Q3", "q": "为什么余弦相似度必须除以范数？", "gold": [
        {"file": "RAG后四环·向量化·检索·组装·生成.md", "must_contain": "只看方向"},
    ]},
    {"id": "Q4", "q": "为什么overlap不能设得很大？", "gold": [
        {"file": "RAG系统课·第1讲.md", "must_contain": "不能设得很大"},
    ]},
    {"id": "Q5", "q": "分块器有几道校验？分别是什么？", "gold": [
        {"file": "项目README.md", "must_contain": "分块长度合规"},
        {"file": "项目README.md", "must_contain": "重叠长度一致性校验"},
        {"file": "分块位置与可溯源.md", "must_contain": "重叠一致性"},
        {"file": "分块与往返校验的实现.md", "must_contain": "分块长度合规"},
    ]},
    {"id": "Q6", "q": "语料加载层有几道哨兵？分别防什么？", "gold": [
        {"file": "元数据过滤与哨兵.md", "must_contain": "名单里的目录"},
        {"file": "上下文检索·让块带着出处进向量.md", "must_contain": "EXCLUDE_DIRS"},
    ]},
    {"id": "Q7", "q": "prev[-0:]为什么会出错？", "gold": [
        {"file": "RAG六步链路·全局地图.md", "must_contain": "整个前一块"},
    ]},
    {"id": "Q8", "q": "删掉Observation那条tool消息，为什么第一反应是API报错？", "gold": [
        {"file": "tool消息的契约.md", "must_contain": "配对缺失"},
        {"file": "ReAct循环搬进图-while的两半.md", "must_contain": "未执行"},
        {"file": "Agent记忆与终止·面试话术.md", "must_contain": "tool_call_id"},
    ]},
    {"id": "Q9", "q": "done字段为什么要用not？它怎么控制循环？", "gold": [
        {"file": "列表与元组.md", "must_contain": "not reply.get"},
        {"file": "ReAct循环.md", "must_contain": "done = not"},
        {"file": "九月总复习体检卷.md", "must_contain": "done"},
    ]},
    {"id": "Q10", "q": "checkpointer存的是最新一张快照还是全部？", "gold": [
        {"file": "checkpointer的两层存储.md", "must_contain": "全部留存"},
        {"file": "checkpointer记忆持久化.md", "must_contain": "每一步之后"},
    ]},
]


def check_gold_alive(docs):
    """gold 里的文件必须在当前语料里，否则该题永远 FAIL 且看不出原因（考卷过期）。"""
    alive = {Path(d["path"]).name for d in docs}
    dead = [(c["id"], g["file"]) for c in EVAL_SET for g in c["gold"] if g["file"] not in alive]
    if dead:
        print("⚠️【考卷过期】下面这些 gold 文件已不在语料里：")
        for qid, f in dead:
            print(f"   {qid}  {f}")
    return dead


def _match_chunk(chunk, gold_list):
    """块级判定：文件名对 + 正文含 must_contain（两层，缺一层就会假绿）。"""
    chunk_filename = Path(chunk.pos.source).name
    chunk_content = chunk.pos.text.strip()

    for gold_item in gold_list:
        if chunk_filename != gold_item["file"]:
            continue
        keyword = gold_item.get("must_contain", "").strip()
        if not keyword:
            return True
        if keyword in chunk_content:
            return True

    return False


def _doc_hit(chunk, gold_list):
    """只看文件名 —— 保留这个指标用于跟历史数字对比。"""
    fname = Path(chunk.pos.source).name
    return any(fname == g["file"] for g in gold_list)


def compute_metrics(rows):
    n = len(rows)
    doc_hit5 = sum(1 for r in rows if r["doc_hit@5"])
    hit5 = sum(1 for r in rows if r["hit@5"])
    ranks = [r["rank"] for r in rows if r["rank"]]
    mrr = sum(1.0 / x for x in ranks) / n if n else 0.0
    return {
        "doc_recall@5": round(doc_hit5 / n, 3) if n else 0,
        "chunk_hit@5": round(hit5 / n, 3) if n else 0,
        "mrr": round(mrr, 3),
        "avg_rank_hit": round(sum(ranks) / len(ranks), 2) if ranks else 0.0,
    }


def make_searcher():
    """测线上同一套检索实现（Chroma），换实现只改这个函数。"""
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def search(query, top_k):
        return search_as_chunks(query, model, top_k=top_k)

    return search


def print_table(rows):
    print()
    print(f"{'题':<5}{'rank':<8}{'doc@5':<8}{'hit@5':<8}")
    print("-" * 32)
    for r in rows:
        rank = r["rank"] if r["rank"] else f">{K_RETRIEVE}"
        print(f"{r['id']:<5}{str(rank):<8}{'✓' if r['doc_hit@5'] else '✗':<8}"
              f"{'✓' if r['hit@5'] else '✗':<8}")


def save_run(payload):
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare_last_run(payload):
    """与上一次运行对比，并给出整体结论（有效 / 正向 / 无影响 / 负向）。"""
    files = sorted(RUNS_DIR.glob("*.json"))
    if len(files) < 2:
        print("\n【对比】还没有上一次记录（这是第一次跑）")
        return
    prev = json.loads(files[-2].read_text(encoding="utf-8"))
    print(f"\n【对比】上一次 = {files[-2].name}")
    for k, v in payload["指标"].items():
        old = prev["指标"].get(k, 0)
        arrow = "↑" if v > old else ("↓" if v < old else "=")
        print(f"   {k:<14} {old} → {v}  {arrow}")

    old_hits = {r["id"]: r["hit@5"] for r in prev.get("每题", [])}
    better = [r["id"] for r in payload["每题"] if r["hit@5"] and not old_hits.get(r["id"], False)]
    worse = [r["id"] for r in payload["每题"] if (not r["hit@5"]) and old_hits.get(r["id"], False)]
    print(f"   变好的题：{better or '（无）'}")
    print(f"   变坏的题：{worse or '（无）'}")

    curr_hit = payload["指标"]["chunk_hit@5"]
    prev_hit = prev["指标"].get("chunk_hit@5", 0)

    if curr_hit > prev_hit and len(worse) == 0:
        conclusion = "✅ 本次改动有效：核心命中率提升，无题目退化"
    elif curr_hit >= prev_hit and len(better) > len(worse):
        conclusion = "⚠️ 本次改动整体正向：收益大于损失，关注退化题目"
    elif curr_hit == prev_hit and len(better) == 0 and len(worse) == 0:
        conclusion = "➖ 本次改动无影响：指标与题目均无变化"
    elif curr_hit < prev_hit or len(worse) > len(better):
        conclusion = "❌ 本次改动负向：核心指标下降或退化题目更多，建议回滚"
    else:
        conclusion = "➖ 本次改动影响不明确"

    print(f"\n   【整体结论】{conclusion}")


def main():
    t0 = datetime.now()

    docs, errors, bom = load_docs()
    print(f"【语料】入选 {len(docs)} 篇  读取失败 {len(errors)} 篇  BOM {bom}")
    check_gold_alive(docs)

    col = _get_collection(DB_PATH, COLLECTION)
    n_chunks = col.count()
    print(f"【向量库】{n_chunks} 条  ← {DB_PATH}")

    search = make_searcher()

    rows = []
    for case in EVAL_SET:
        hits = search(case["q"], K_RETRIEVE)
        rank = None
        for i, ch in enumerate(hits, 1):
            if _match_chunk(ch, case["gold"]):
                rank = i
                break
        rows.append({
            "id": case["id"],
            "q": case["q"],
            "rank": rank,
            "doc_hit@5": any(_doc_hit(c, case["gold"]) for c in hits[:K_REPORT]),
            "hit@5": any(_match_chunk(c, case["gold"]) for c in hits[:K_REPORT]),
        })

    m = compute_metrics(rows)
    payload = {
        "时间": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "语料": {"篇数": len(docs), "块数": n_chunks},
        "检索": {"实现": "chroma", "K_retrieve": K_RETRIEVE, "K_report": K_REPORT},
        "指标": m,
        "每题": rows,
    }
    print_table(rows)
    print()
    for k, v in m.items():
        print(f"  {k:<16}{v}")
    path = save_run(payload)
    print(f"\n【存档】{path.name}")
    compare_last_run(payload)
    print(f"\n【耗时】{(datetime.now() - t0).total_seconds():.1f} 秒")


if __name__ == "__main__":
    main()
