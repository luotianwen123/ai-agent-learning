# -*- coding: utf-8 -*-
"""
eval_rag.py —— RAG 检索的「测评线」

════════════════════════════════════════════════════════════════════
这份文件的目的（先读这段，再看代码）
────────────────────────────────────────────────────────────────────
它只回答一个问题：
    我改了检索之后，到底变好了还是变坏了？用数字说。

所以它必须满足三条，缺一条就没意义：
  1. 题目固定      —— 每次跑同一套题，否则数字不可比
  2. 判定是机器做的 —— 人手判 = 会骗自己
  3. 结果能存档对比 —— 存成 JSON，下次跑完跟上次比

════════════════════════════════════════════════════════════════════
四层结构（改代码前先想清楚自己在改哪一层）
────────────────────────────────────────────────────────────────────
  ① 数据层  EVAL_SET          ：题 + gold（文件 + 该答案里必含的关键词）
  ② 检索层  make_searcher()   ：只负责"给 query，返回按相似度排好的块"
  ③ 指标层  _match_chunk / compute_metrics ：只负责算数，不打印
  ④ 报告层  print_table / save_run / compare_last_run ：输出 + 存档 + 跟上次比

层与层之间只传数据、不互相 print。这样以后把 Chroma 换成 Reranker，
只动 ②，其余三层一行都不用改。

════════════════════════════════════════════════════════════════════
【你要自己填的 3 处】——都已标成 TODO，搜 "TODO" 就能找到
────────────────────────────────────────────────────────────────────
  TODO ①  每道题的 must_contain 关键词     ← 必须你填，只有你知道答案在哪句话
  TODO ②  _match_chunk 的判定逻辑         ← 先照着跑通，然后用自己的话重写
  TODO ③  compare_last_run 的"变好/变坏"   ← 自己定义什么叫变好
"""

import json
import os
from datetime import datetime
from pathlib import Path

# 必须在 sentence_transformers 之前 —— 否则它会去 huggingface.co 查更新，国内直连超时
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

from notes_loader import load_docs
from chroma_store import search_as_chunks, _get_collection, DB_PATH, COLLECTION

# ────────────────────────────────────────────────────────────────
# ① 数据层
# ────────────────────────────────────────────────────────────────
# K_RETRIEVE：先从库里捞多少条回来（捞 10 条，才能算出"排第几"）
# K_REPORT  ：对外报的 Recall@K / Hit@K（面试说"Recall@5"才标准）
K_RETRIEVE = 10
K_REPORT = 5

RUNS_DIR = Path(__file__).with_name("eval_runs")   # 每次运行存一份 JSON

# 每道题：q = 问题；gold = 一组"正确答案在哪里"的候选
#   file          ：哪个文件
#   must_contain  ：那句话里必然出现的一个关键词（空字符串 = 退化成"只看文件"）
#
# ⭐ TODO ①（最重要，必须你填）
#   为什么必须加 must_contain：
#     只判文件名的话，"召回了正确文件、但那块只是 frontmatter / 标题行"
#     也会被算成命中 —— 这就是你之前看到的"90% 全绿但检索是垃圾"。
#   怎么填：
#     打开那篇笔记，找到真正回答这句话的那一句，挑一个【只在这句/这一节出现】的词。
#     别挑 "RAG" "Agent" 这种全文到处都是的词（等于没加约束）。
#     填完跑一次：如果 hit 从 9/10 掉到 6/10 —— 那不是退步，是终于测出真相。
EVAL_SET = [
    {"id": "Q1", "q": "什么是RAG？", "gold": [
        {"file": "RAG六步链路·全局地图.md", "must_contain": ""},
        {"file": "RAG系统课·第1讲.md", "must_contain": ""},
        {"file": "Agent术语中英对照表.md", "must_contain": ""},
    ]},
    {"id": "Q2", "q": "为什么说切分是RAG的命门？", "gold": [
        {"file": "RAG系统课·第1讲.md", "must_contain": ""},
    ]},
    {"id": "Q3", "q": "为什么余弦相似度必须除以范数？", "gold": [
        {"file": "RAG后四环·向量化·检索·组装·生成.md", "must_contain": ""},
    ]},
    {"id": "Q4", "q": "为什么overlap不能设得很大？", "gold": [
        {"file": "RAG系统课·第1讲.md", "must_contain": ""},
    ]},
    {"id": "Q5", "q": "分块器有几道校验？分别是什么？", "gold": [
        {"file": "分块位置与可溯源.md", "must_contain": ""},
        {"file": "分块与往返校验的实现.md", "must_contain": ""},
        {"file": "项目README.md", "must_contain": ""},
    ]},
    {"id": "Q6", "q": "语料加载层有几道哨兵？分别防什么？", "gold": [
        {"file": "元数据过滤与哨兵.md", "must_contain": ""},
        {"file": "上下文检索·让块带着出处进向量.md", "must_contain": ""},
    ]},
    {"id": "Q7", "q": "prev[-0:]为什么会出错？", "gold": [
        {"file": "RAG六步链路·全局地图.md", "must_contain": ""},
        {"file": "RAG系统课·第1讲.md", "must_contain": ""},
    ]},
    {"id": "Q8", "q": "删掉Observation那条tool消息，为什么第一反应是API报错？", "gold": [
        {"file": "tool消息的契约.md", "must_contain": ""},
        {"file": "ReAct循环.md", "must_contain": ""},
    ]},
    {"id": "Q9", "q": "done字段为什么要用not？它怎么控制循环？", "gold": [
        {"file": "ReAct循环.md", "must_contain": ""},
    ]},
    {"id": "Q10", "q": "checkpointer存的是最新一张快照还是全部？", "gold": [
        {"file": "checkpointer的两层存储.md", "must_contain": ""},
        {"file": "checkpointer记忆持久化.md", "must_contain": ""},
    ]},
]


def check_gold_alive(docs):
    """gold 里的文件必须在当前语料里 —— 否则这题永远 FAIL，而你查不出原因。

    为什么要有这一步：笔记会改名、会搬家、会删。gold 写的是旧名字时，
    指标只会默默变差，看起来像"检索坏了"，其实是【考卷过期了】。
    这类"看起来像 A 其实是 B"的问题，必须在开跑前就喊出来。
    """
    alive = {Path(d["path"]).name for d in docs}
    dead = [(c["id"], g["file"]) for c in EVAL_SET for g in c["gold"] if g["file"] not in alive]
    if dead:
        print("⚠️【考卷过期】下面这些 gold 文件已不在语料里：")
        for qid, f in dead:
            print(f"   {qid}  {f}")
        print("   → 这题会永远 FAIL，先去改 gold，或把文件找回来")
    return dead


# ────────────────────────────────────────────────────────────────
# ③ 指标层（先定义"怎么判"，再跑；顺序别反）
# ────────────────────────────────────────────────────────────────
def _match_chunk(chunk, gold_list):
    """判断"这一块"是不是某个候选答案。

    规则（两层，缺一层就会误判）：
      1. 文件名要对该 —— 只判文件名 = 假绿
      2. 正文含 must_contain（若填了）—— 这一步才真正排除 frontmatter / 标题行

    TODO ②（先跑通，然后用自己的话重写这一段）
      写之前先回答自己：两块来自同一个文件，一块能答、一块只是标题行，
      这里靠什么把它们区分开？如果你答不上来，说明这段你还没读懂。
    """
    fname = Path(chunk.pos.source).name
    body = chunk.pos.text
    for g in gold_list:
        if fname != g["file"]:
            continue
        kw = g.get("must_contain", "")
        if kw and kw not in body:
            continue
        return True
    return False


def _doc_hit(chunk, gold_list):
    """只看文件名 —— 保留这个指标，是为了跟历史数字对比（旧口径）。"""
    fname = Path(chunk.pos.source).name
    return any(fname == g["file"] for g in gold_list)


def compute_metrics(rows):
    """只算数、不打印。以后要出别的报表，直接复用这个函数。"""
    n = len(rows)
    doc_hit5 = sum(1 for r in rows if r["doc_hit@5"])
    hit5 = sum(1 for r in rows if r["hit@5"])
    ranks = [r["rank"] for r in rows if r["rank"]]
    mrr = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n if n else 0.0
    return {
        "doc_recall@5": round(doc_hit5 / n, 3) if n else 0,
        "chunk_hit@5": round(hit5 / n, 3) if n else 0,
        "mrr": round(mrr, 3),
        "avg_rank_hit": round(sum(ranks) / len(ranks), 2) if ranks else 0.0,
    }


# ────────────────────────────────────────────────────────────────
# ② 检索层：换实现只改这一个函数
# ────────────────────────────────────────────────────────────────
def make_searcher():
    """返回 search(query, top_k) -> list[ChunkItem]（已按相似度排好序）。

    为什么用 Chroma（而不是内存暴力余弦）：
      服务（FAST_API）用的是 Chroma。**测评必须测线上那一套**，
      否则你测的是一个已经没人用的实现 —— 数字再好看也没意义。
    """
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def search(query, top_k):
        return search_as_chunks(query, model, top_k=top_k)

    return search


# ────────────────────────────────────────────────────────────────
# ④ 报告层
# ────────────────────────────────────────────────────────────────
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
    """跟上一次运行对比 —— 这是整条测评线最值钱的一段。

    没有它，你只能说"现在是 9/10"；
    有了它，你能说"从 8/10 变成 9/10，是这次改动带来的"。
    """
    files = sorted(RUNS_DIR.glob("*.json"))
    if len(files) < 2:
        print("\n【对比】还没有上一次记录（这是第一次跑），下次再对比")
        return
    prev = json.loads(files[-2].read_text(encoding="utf-8"))
    print(f"\n【对比】上一次 = {files[-2].name}")
    for k, v in payload["指标"].items():
        old = prev["指标"].get(k, 0)
        arrow = "↑" if v > old else ("↓" if v < old else "=")
        print(f"   {k:<14} {old} → {v}  {arrow}")

    # TODO ③（自己定义"变好/变坏"，然后把它写清楚）
    #   提示：一次改动几乎不会让所有题一起变好。
    #   更有信息量的是「哪些题从 ✗ 变 ✓、哪些从 ✓ 变 ✗」。
    #   想清楚：指标涨了、但某道原本对的题变错了 —— 这次改动算成功吗？
    old_hits = {r["id"]: r["hit@5"] for r in prev.get("每题", [])}
    better = [r["id"] for r in payload["每题"] if r["hit@5"] and not old_hits.get(r["id"], False)]
    worse = [r["id"] for r in payload["每题"] if (not r["hit@5"]) and old_hits.get(r["id"], False)]
    print(f"   变好的题：{better or '（无）'}")
    print(f"   变坏的题：{worse or '（无）'}")


def main():
    t0 = datetime.now()

    # ── 数据层：先看考卷有没有过期 ──
    docs, errors, bom = load_docs()
    print(f"【语料】入选 {len(docs)} 篇  读取失败 {len(errors)} 篇  BOM {bom}")
    check_gold_alive(docs)

    # 向量库：直接用建好的库（不重新分块、不重新 encode）
    col = _get_collection(DB_PATH, COLLECTION)
    n_chunks = col.count()
    print(f"【向量库】{n_chunks} 条  ← {DB_PATH}")

    search = make_searcher()

    # ── 主循环：平的一层，不嵌套 ──
    rows = []
    for case in EVAL_SET:
        hits = search(case["q"], K_RETRIEVE)          # 一次检索，捞 10 条

        rank = None
        for i, ch in enumerate(hits, 1):               # 找"第几块能答"
            if _match_chunk(ch, case["gold"]):
                rank = i
                break

        rows.append({
            "id": case["id"],
            "q": case["q"],
            "rank": rank,                                          # 未命中 = None
            "doc_hit@5": any(_doc_hit(c, case["gold"]) for c in hits[:K_REPORT]),
            "hit@5": any(_match_chunk(c, case["gold"]) for c in hits[:K_REPORT]),
        })

    # ── 汇总 + 存档 + 对比 ──
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
