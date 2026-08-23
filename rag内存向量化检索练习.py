from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
@dataclass
class ChunkItem:
    text: str
    vector: list[float]
def simple_chunk_split(text: str, chunk_size:int=50, overlap:int=0):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
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
    memory_vector_store: list[ChunkItem] = []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    chunks = simple_chunk_split(demo_doc, chunk_size=50, overlap=10)
    for one_chunk in chunks:
        vec = model.encode(one_chunk)
        item = ChunkItem(text=one_chunk, vector=vec.tolist())
        memory_vector_store.append(item)
    print(f"向量库构建完成，chunk数量：{len(memory_vector_store)}")

    user_query = "什么是RAG"
    res = retrieve(user_query, model, memory_vector_store, top_k=2)
    print("\n检索召回片段：")
    for s in res:
        print(s)

    if len(res) > 0:
        context_text = "\n\n".join(res)
    else:
        context_text = "无相关参考资料"

    prompt = build_prompt_simple(context_text, user_query)
    print("\n组装完成Prompt：")
    print(prompt)
