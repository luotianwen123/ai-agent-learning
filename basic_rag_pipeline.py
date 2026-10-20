"""
一个轻量级 RAG（检索增强生成）示例：
将知识文档分块 -> 用 bge-small-zh 向量化 -> 按余弦相似度检索 TopK 片段 ->
按 token 预算裁剪上下文后组装 prompt，调用 DeepSeek Chat API 生成回答。

语料库来源：自学的知识整理的obsidian仓库

依赖安装：
    pip install sentence-transformers tiktoken requests numpy python-dotenv

运行：
    1. 在 .env 中填入 OPENAI_API_KEY（DeepSeek 开放平台申请的 API Key）
    2. python basic_rag_pipeline.py
    3. 首次运行会自动下载向量模型 BAAI/bge-small-zh-v1.5

返回值：list[ChunkPosition]（每块文本 + 在原文中的区间）
        注意：_raw=True 时（供内部递归使用）返回 list[str]（无重叠的原始块）
"""
import os
import re

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from sentence_transformers import SentenceTransformer
from notes_loader import load_docs
from dataclasses import dataclass,replace
import numpy as np
import tiktoken
import requests
import json

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY 未配置，请在 .env 文件中填写")
BASE_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

# 全局统一分隔符：分块逻辑、边界校验共用，避免两处定义不一致
SEPARATORS = ["\n\n", "。", "\n", "|",  "；", "：", " "]


@dataclass(frozen=True) #frozen=True 的意思是这个对象一旦造出来就不能改
class ChunkPosition:
    text: str      # 这一块的实际字符串（含重叠前缀）	要拿去编码、拼 prompt
    start:int      #这块在原文中的起点（字符下标）	回溯定位 —— "这段是哪来的"
    end:int        #这块新增正文的终点	配合 start 算出区间 [start, end)
    actual_overlap:int #实际前置了几个字
    truncated:bool #是否被 max_chunk_size 截断过	有这个标记才知道"这块不完整"
    header:str=""  # 这块属于哪一节—— 递归里只有片段、算不出来
    source:str=""  # 哪份文件 —— 分块函数不知道，由上层填
    index:int=-1   # 同一文档内的块序号 —— 同上

@dataclass
class ChunkItem:
    pos:ChunkPosition
    vector: list[float]


def recursive_split(text: str, max_chunk_size: int, overlap: int, _raw: bool = False) -> list[ChunkPosition]:
    # 参数校验只在最外层跑，递归进去跳过
    if not _raw:
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size={max_chunk_size}必须为正整数")
        if overlap < 0:
            raise ValueError(f"overlap={overlap}不能为负数")
        if overlap >= max_chunk_size:
            raise ValueError(f"overlap={overlap}不能大于等于max_chunk_size={max_chunk_size}")

    # 原始无重叠块上限：预留重叠空间，拼接后刚好等于max_chunk_size
    raw_chunk_max = max_chunk_size - overlap

    all_chunks = []
    found_split = False
    for sep in SEPARATORS:
        # 【核心修复】在前raw_chunk_max范围内，从右往左找最后一个分隔符
        # 保证左半块不超长度上限，且尽可能在语义边界断开
        split_pos = text[:raw_chunk_max].rfind(sep)
        if split_pos != -1:
            left_part = text[:split_pos + len(sep)]
            right_part = text[split_pos + len(sep):]
            if len(left_part) == len(text):
                continue
            # 传 _raw=True 拿原始分片，避免每层都叠一次重叠
            left_list = recursive_split(left_part, max_chunk_size, overlap, _raw=True)
            right_list = recursive_split(right_part, max_chunk_size, overlap, _raw=True)
            all_chunks.extend(left_list)
            all_chunks.extend(right_list)
            found_split = True
            break

    if not found_split:
        i = 0
        while i < len(text):
            chunk = text[i:i + raw_chunk_max]
            all_chunks.append(chunk)
            i += raw_chunk_max

    buffer = []
    buffer_len = 0
    merged_chunks = []
    for chunk in all_chunks:
        chunk_len = len(chunk)
        # 新增缓冲区非空判断，避免空块；阈值统一用原始块上限
        if buffer and buffer_len + chunk_len > raw_chunk_max:
            merged_chunks.append("".join(buffer))
            buffer = [chunk]
            buffer_len = chunk_len
        else:
            buffer.append(chunk)
            buffer_len += chunk_len
    if buffer_len != 0:
        merged_chunks.append("".join(buffer))

    # 递归内部直接返回，重叠只在最外层拼一次
    if _raw:
        return merged_chunks

    positions =[]   # ← 装每块的位置，循环里往里放
    cursor = 0  # ← 游标：读头当前在原文的哪个位置

    for idx, chunk in enumerate(merged_chunks):
        truncated = False  # ← 先假定"没被截断"

        if idx == 0:
            actual_overlap = 0
            new_chunk = chunk
        else:
            prev = merged_chunks[idx - 1]
            if overlap>0:
                prev_tail = prev[-overlap:]
            else:
                prev_tail = ""
            new_chunk = prev_tail  + chunk
            actual_overlap =  len(prev_tail)    # ← 本块开头【实际】被前置了几字
            if len(new_chunk) > max_chunk_size:
                new_chunk = new_chunk[:max_chunk_size]
                truncated = True  # ← 进了这条分支，说明我们的文本被截断


        # —— 记录位置（今天只收集，不改返回值）——
        end_of_new = cursor+len(chunk)   # ← 本块【新增正文】的终点
        start = cursor-actual_overlap  # ← 整块区间的起点
        positions.append(ChunkPosition(
            text=new_chunk,
            start=start,
            end=end_of_new,
            actual_overlap=actual_overlap,
            truncated=truncated,
            source="",  # 哪份文件由上层填
            index=idx,  # 本块在本文档里的第几块
        ))
        cursor = end_of_new  #  推进读头

    assert positions[0].start == 0, f"第 0 块起点应为 0，实际 {positions[0].start}"
    assert len(positions) == len(merged_chunks), "有块没被记录"

    # 只打印：前 3 块 + 第 39 块 + 最后 1 块
    for p in positions[:3] + positions[39:40] + positions[-1:]:
        print(f"块{p.index} 原文[{p.start},{p.end}) 实际重叠{p.actual_overlap} "
            f"截断{p.truncated} 尾20字={p.text[-20:]!r}")
    return positions  #只回位置对象


def row_num(text:str,offset:int)->int:
    return text[:offset].count("\n")+1


def subsection_location(text:str,offset:int)->str:
    section="(无标题区)"
    for m in re.finditer(r"^#{1,6} .*$", text,re.M):
        if m.start() <= offset:
            section = m.group().strip()
        else:
            break
    return section

def check_max_chunk(chunks: list[str], max_chunk_size: int,) -> tuple[bool, int, str]:
    """
    校验所有分块的字符长度是否不超过设定上限
    :param chunks: 待校验文本块列表
    :param max_chunk_size: 单块最大字符数上限
    :return: (是否校验通过, 最长块字符数量, 最长块文本内容)
    """
    max_chunk_num = max(len(w) for w in chunks)
    max_chunk_text = ""
    for text in chunks:
        if len(text) == max_chunk_num:
            max_chunk_text = text
    if max_chunk_num > max_chunk_size:
        return False, max_chunk_num, max_chunk_text
    return True, max_chunk_num, max_chunk_text

def check_boundary(chunks: list[str], separators: list[str]) -> tuple[int, int, list[int]]:
    """
    校验分块结尾是否为合法分隔符（排除最后一块，最后一块允许不完整）
    :return: (不合格块数量, 已检查块总数, 不合格块下标列表)
    """
    bad_index = []
    checked = 0
    for i, chunk in enumerate(chunks[:-1]):
        checked += 1
        # 修复endswith的坑：必须传元组，不能传列表，否则报TypeError
        if not chunk.endswith(tuple(separators)):
            bad_index.append(i)
    bad_count = len(bad_index)
    return bad_count, checked, bad_index

def first_diff_index(text_a: str, text_b: str) -> int:
    """
    找出两个字符串首个不同的字符下标
    :return: 首个差异下标；完全相同返回 -1；前缀一致仅长度不同时，返回较短串长度
    """
    for i, (ca, cb) in enumerate(zip(text_a, text_b)):
        if ca != cb:
            return i
    if len(text_a) != len(text_b):
        return min(len(text_a), len(text_b))
    return -1

def format_first_diff(restored: str, original: str, diff_index: int) -> str:
    """
    把首个差异位置渲染成可直接读的说明：截取差异点前后各 8 字对照，便于肉眼定位
    :param restored: 还原出的文本
    :param original: 原文
    :param diff_index: first_diff_index 的返回值，-1 表示无差异
    :return: 形如「第 47 字：还原「…abc」/ 原文「…abd」」；无差异时返回「无差异」
    """
    if diff_index < 0:
        return "无差异"
    span = 8
    start = max(0, diff_index - span)
    # 差异点落在某一侧末尾之外：本质是「谁更长」，直接点名多出 / 缺少的内容
    if diff_index >= len(original):
        return f"第 {diff_index} 字起：原文已结束，还原仍多出「{restored[diff_index:diff_index + span]}」"
    if diff_index >= len(restored):
        return f"第 {diff_index} 字起：还原已结束，原文尚余「{original[diff_index:diff_index + span]}」"
    return f"第 {diff_index} 字：还原「{restored[start:diff_index + span]}」/ 原文「{original[start:diff_index + span]}」"

def check_no_loss(chunks: list[str], original: str, overlaps: list[int]) -> tuple[bool, int, int, str]:
    """
    去掉每个块的前缀重叠后拼接还原，与原文比对验证无信息丢失
    :param chunks: 带重叠的分块结果
    :param original: 原始完整文本
    :param overlaps: 每一块【实际】的前置重叠字数（第 i 块用 overlaps[i]）
    :return: (是否完全一致, 还原后文本长度, 原文长度, 首个差异说明，一致时为「无差异」)
    """
    # 【已知问题·overlap=0】prev[-0:] == prev[0:] 会取到整个前一块 → 块1 开头多塞块0 全文
    #   实测：overlap=0 → 还原 263 / 原文 230（多 33 字）
    #   同时 check_no_loss 的 chunk[overlap:] 在 overlap=0 时也不剁，还原式自己失效
    #   校验网格覆盖：mcs<=0 ✓ / overlap<0 ✓ / overlap>=mcs ✓ / **overlap==0 是空格**
    if not chunks:
        restored = ""
    else:
        restored_parts = [chunks[0]]
        for i in range(1,len(chunks)):
            restored_parts.append(chunks[i][overlaps[i]:])
        restored = "".join(restored_parts)

    is_equal = (restored == original)
    # 补首个差异说明：原先只回两个长度，碰到「长度相同但内容不同」会报出两个一模一样的数字，看不出问题
    diff_desc = format_first_diff(restored, original, first_diff_index(restored, original))
    return is_equal, len(restored), len(original), diff_desc

def check_overlap_consistency(chunks: list[str], overlaps: list[int]) -> tuple[bool, int, list[int]]:
    """
    校验相邻块之间重叠区域一致性
    规则：第i块(A)末尾overlap个字符，必须等于第i+1块(B)开头overlap个字符
    :param chunks: 带重叠的文本分块列表
    :param overlaps: 每一块【实际】的前置重叠字数（第 i+1 块前置的是 overlaps[i+1] 个字）
    :return: (是否全部重叠合法, 错误的相邻块对总数, 出错块对下标列表)
        注意：返回的下标i代表 chunks[i] 和 chunks[i+1] 这一对校验失败
    """
    bad_pair_idx = []
    for i, chunk_a in enumerate(chunks[:-1]):
        chunk_b = chunks[i+1]
        n = overlaps[i+1]
        if n == 0:
            continue
        if chunk_a[-n:] != chunk_b[:n]:
            bad_pair_idx.append(i)
    bad_count = len(bad_pair_idx)
    all_ok = (bad_count == 0)
    return all_ok, bad_count, bad_pair_idx

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve(query: str, model, vector_store: list[ChunkItem], top_k: int = 2) -> list[ChunkItem]:
    top_k = min(top_k, len(vector_store))
    q_emb = model.encode(query)
    score_list = []
    for item in vector_store:
        score = cosine_similarity(q_emb, item.vector)
        score_list.append((score, item))
    score_list.sort(key=lambda x: x[0], reverse=True)
    recall_result = score_list[:top_k]
    top_chunks = [text for score, text in recall_result]
    return top_chunks


def calc_available_chunk_quota(model_max_window: int,
                               system_prompt: str,
                               user_query: str,
                               tokenizer,
                               reserve_output_token: int) -> int:
    sys_tokens = len(tokenizer.encode(system_prompt))
    query_tokens = len(tokenizer.encode(user_query))
    available_chunk_token = model_max_window - sys_tokens - query_tokens - reserve_output_token
    if available_chunk_token <= 0:
        return 0
    return available_chunk_token


def clip_context_by_max_token(chunk_list: list[str], token_limit: int, tokenizer) -> str:
    total_tokens = 0
    keep_chunks = []
    for one_chunk in chunk_list:
        current_chunk_token = len(tokenizer.encode(one_chunk))
        if total_tokens + current_chunk_token > token_limit:
            break
        total_tokens += current_chunk_token
        keep_chunks.append(one_chunk)
    safe_context = "\n".join(keep_chunks)
    return safe_context


def build_rag_prompt(system_prompt: str,
                     safe_context: str,
                     user_query: str) -> str:
    parts = [system_prompt]
    if safe_context.strip():
        parts.append("\n【参考文档】")
        parts.append(safe_context)
    parts.append(f"\n用户问题:{user_query}")
    full_prompt = "\n".join(parts)
    return full_prompt


def llm_chat(
        bearer_key: str,
        model_name: str,
        prompt_text: str,
        base_url: str,
        max_output_tokens: int = 512
) -> str:
    headers = {
        "Authorization": f"Bearer {bearer_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_output_tokens
    }
    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            raise RuntimeError("接口429限流：请求过于频繁")
        if resp.status_code != 200:
            raise RuntimeError(f"接口请求失败，status_code:{resp.status_code}, {resp.text}")
        try:
            resp_json = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"返回JSON解析失败：{str(e)}")
        result_text = resp_json["choices"][0]["message"]["content"]
        return result_text
    except requests.exceptions.RequestException as e:
        # 工具层只做异常包装，交给业务层决定
        raise RuntimeError(f"网络请求异常：{str(e)}")


# ── 语料来源：改成"真从笔记里读出来的正文" ──
def build_corpus(docs: list[dict], max_chunk_size: int, overlap: int):
    """第一段（文档级）：每篇各自分块 → 补 header/source → 逐篇跑四道校验

    :return: (all_positions, report, failures)
             all_positions = 全库所有块（含"跳过校验"那篇的块）
             report        = [(文件名, "通过"/"跳过"/"失败", 细节), ...]
             failures      = [(文件名, [原因, ...]), ...]  （非空时由调用方决定要不要 raise）
    """
    all_positions: list[ChunkPosition] = []
    report, failures = [], []

    for d in docs:
        name = Path(d["path"]).name
        text = d["text"]

        # frontmatter 清理：用【空格占位】而不是删掉。
        # 删了会让本文件所有 start/end/行号整体前移，出处全错且不报错（静默错）；
        # 空格占位长度不变，位置对得上。frontmatter 主题词密度高，会抢检索的高分位。
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = " " * (end + 4) + text[end + 4:]

        ps = recursive_split(text, max_chunk_size=max_chunk_size, overlap=overlap)
        # header 用【这一篇的】text 算；source 一起补（否则 50 篇混完就分不清来源）
        ps = [replace(p, header=subsection_location(text, p.start), source=d["path"]) for p in ps]
        assert all(p.header for p in ps), f"{name}: 有块的 header 是空的"

        all_positions.extend(ps)          # ★ 先入库 —— 后面走哪个分支都不会漏

        ch = [p.text for p in ps]
        ov = [p.actual_overlap for p in ps]

        if len(ch) < 2:                   # 单块文档：没有相邻对可校验 → 跳过，但必须上报
            report.append((name, "跳过", "只分出 1 块，无相邻对可校验"))
            continue

        ok1, max_num, _ = check_max_chunk(ch, max_chunk_size=max_chunk_size)
        bad_cnt, checked_num, bad_idx = check_boundary(ch, SEPARATORS)
        ok2 = (bad_cnt == 0) and (checked_num > 0)
        ok3, bad_pairs, bad_pair_idx = check_overlap_consistency(ch, ov)
        ok4, restored_len, origin_len, diff_desc = check_no_loss(ch, text, ov)

        if ok1 and ok2 and ok3 and ok4:
            report.append((name, "通过", f"{len(ch)} 块 · 最长 {max_num}"))
            continue

        why = []
        if not ok1: why.append(f"块超长 {max_num}>{max_chunk_size}")
        if not ok2: why.append(f"边界不合格 {bad_cnt}/{checked_num} 坏块{bad_idx[:3]}")
        if not ok3: why.append(f"重叠不匹配 {bad_pairs} 对 {bad_pair_idx[:3]}")
        if not ok4: why.append(f"还原 {restored_len}/{origin_len} {diff_desc}")
        report.append((name, "失败", "；".join(why)))
        failures.append((name, why))

    return all_positions, report, failures


if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_chunk_size, overlap = 150, 30

    # ===== 第一段：文档级 =====
    docs, errors, bom_files = load_docs()
    assert docs, "语料库一篇都没读进来，先单独跑 python notes_loader.py 排查"
    path2text = {d["path"]: d["text"] for d in docs}  # 算行号要用（source → 正文）
    print(f"【语料载入】共 {len(docs)} 篇 | 带 BOM {bom_files} 篇"
          + (f" | 跳过 {len(errors)} 篇" if errors else ""))

    all_positions, report, failures = build_corpus(docs, max_chunk_size, overlap)

    n_skip = sum(1 for _, v, _ in report if v == "跳过")
    n_fail = sum(1 for _, v, _ in report if v == "失败")
    print(f"【逐篇校验】{len(report)} 篇 → 通过 {len(report) - n_skip - n_fail} · 跳过 {n_skip} · 失败 {n_fail}")
    for name, verdict, detail in report:
        if verdict != "通过":
            print(f"  {verdict}  {name}  {detail}")
    print(f"【数量对账】全库块数 {len(all_positions)}")
    assert not failures, f"有 {len(failures)} 篇没通过逐篇校验，见上方明细"

    # ===== 第二段：全局级 =====
    chunks = [p.text for p in all_positions]
    vectors = model.encode(chunks, batch_size=32).tolist()  # 顺手修掉 O(n²)
    assert len(vectors) == len(all_positions), "向量数与块数不匹配"
    vector_store = [ChunkItem(pos=p, vector=v) for p, v in zip(all_positions, vectors)]

    query = "什么是RAG？"
    retrieved = retrieve(query, model, vector_store, top_k=3)

    print(f"【检索命中】query = {query!r}")
    for it in retrieved:
        p = it.pos
        t = path2text[p.source]  # ← 用那块自己的文件算行号
        print(f"  · {Path(p.source).name} · 第 {row_num(t, p.start)}–{row_num(t, p.end - 1)} 行 "
              f"· {p.header} · 片段: {p.text[:28]!r}…")

    system_prompt = "你是知识库问答助手，请依据下面参考文档回答用户问题，如果文档没有答案就如实说明，禁止编造幻觉内容。"
    MODEL_MAX_WINDOW = 4096
    RESERVE_OUTPUT_TOKEN = 512

    available_chunk_token = calc_available_chunk_quota(
        model_max_window=MODEL_MAX_WINDOW,
        system_prompt=system_prompt,
        user_query=query,
        tokenizer=tokenizer,
        reserve_output_token=RESERVE_OUTPUT_TOKEN
    )
    if available_chunk_token <= 0:
        print("【警告】可用知识库token配额为0，不会加载任何参考文档片段")

    safe_context = clip_context_by_max_token(
        chunk_list=[it.pos.text for it in retrieved],
        token_limit=available_chunk_token,
        tokenizer=tokenizer
    )
    final_prompt = build_rag_prompt(
        system_prompt=system_prompt,
        safe_context=safe_context,
        user_query=query
    )

    print("====组装完成的Prompt====")
    print(final_prompt)
    try:
        answer = llm_chat(
            bearer_key=API_KEY,
            model_name=MODEL,
            prompt_text=final_prompt,
            base_url=BASE_URL
        )
        print("\n====大模型返回回答====")
        print(answer)
    except RuntimeError as e:
        print(f"【接口调用异常】{e}")
