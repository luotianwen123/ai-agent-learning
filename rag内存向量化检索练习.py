from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class ChunkItem:
    text: str
    vector: list[float]

def recursive_split(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "。", "\n"]

    if len(text) <= max_chunk_size:
        return [text]

    all_chunks = []
    found_split = False

    for sep in separators:
        split_pos = text.rfind(sep)
        if split_pos != -1:
            left_part = text[:split_pos + len(sep)]
            right_part = text[split_pos + len(sep):]

            if len(left_part) == len(text) or len(right_part) == len(text):
                continue

            left_list = recursive_split(left_part, max_chunk_size, overlap)
            right_list = recursive_split(right_part, max_chunk_size, overlap)
            all_chunks.extend(left_list)
            all_chunks.extend(right_list)
            found_split = True
            break

    if not found_split:
        i = 0
        while i < len(text):
            chunk = text[i:i+max_chunk_size]
            all_chunks.append(chunk)
            i += max_chunk_size

    buffer = []
    buffer_len = 0
    merged_chunks = []
    for chunk in all_chunks:
        chunk_len = len(chunk)
        if buffer_len + chunk_len > max_chunk_size:
            merged_chunks.append("".join(buffer))
            buffer = [chunk]
            buffer_len = chunk_len
        else:
            buffer.append(chunk)
            buffer_len += chunk_len

    if buffer_len != 0:
        merged_chunks.append("".join(buffer))

    final_chunks = []
    for i in range(len(merged_chunks)):
        current = merged_chunks[i]
        if i > 0:
            prev_chunk = merged_chunks[i-1]
            overlap_text = prev_chunk[-overlap:]
            current = overlap_text + current
        final_chunks.append(current)

    return final_chunks

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b))

def retrieve(query: str, model, vector_store: list[ChunkItem], top_k=2):
    top_k = min(top_k, len(vector_store))
    q_emb = model.encode(query)
    score_list = []
    for item in vector_store:
        score = cosine_similarity(q_emb, item.vector)
        score_list.append((score, item.text))
    score_list.sort(key=lambda x:x[0], reverse=True)
    recall_result = score_list[:top_k]
    top_chunks = [text for score, text in recall_result]
    return top_chunks

def build_prompt_simple(context:str, query:str)->str:
    prompt_template = """请严格依据下面的参考资料回答用户的问题，如果参考资料没有相关信息就如实说明不知道，不要编造内容。
参考资料：
{context}
用户问题：{query}
回答："""
    final_prompt = prompt_template.format(context=context, query=query)
    return final_prompt

demo_doc = """Agent（智能体）可以自主规划任务，调用工具，读取记忆。
RAG检索增强生成，通过知识库检索，给大模型补充外部资料，减少幻觉。
文本分块是RAG第一步，合理的分块大小直接影响检索效果。分块过大混入无关信息；分块过小丢失完整语义。"""

if __name__ == "__main__":
    model = SentenceTransformer("all‑MiniLM‑L6‑v2")

    chunks = recursive_split(demo_doc, max_chunk_size=150, overlap=20)

    vector_store: list[ChunkItem] = []
    for c in chunks:
        emb = model.encode(c).tolist()
        vector_store.append(ChunkItem(text=c, vector=emb))

    query = "什么是RAG？"
    retrieved_chunks = retrieve(query, model, vector_store, top_k=2)

    context_text = "\n".join(retrieved_chunks)
    final_prompt = build_prompt_simple(context_text, query)
    print(final_prompt)
