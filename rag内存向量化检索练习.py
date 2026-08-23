from sentence_transformers import SentenceTransformer
import numpy as np

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

demo_doc = """Agent（智能体）可以自主规划任务，调用工具，读取记忆。
RAG检索增强生成，通过知识库检索，给大模型补充外部资料，减少幻觉。
文本分块是RAG第一步，合理的分块大小直接影响检索效果。分块过大混入无关信息；分块过小丢失完整语义。"""

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = simple_chunk_split(demo_doc, chunk_size=50, overlap=10)
embeds = model.encode(chunks)

# model也变成入参，函数内部不再读取任何外部全局变量
def retrieve(query: str, model, chunks, embeds, top_k=2):
    top_k = min(top_k, len(chunks))
    q_emb = model.encode(query)
    sim = np.dot(embeds, q_emb)
    nums = np.argsort(sim)[::-1][:top_k]
    top_chunks = [chunks[i] for i in nums]
    return top_chunks
top_chunks = ["片段A","片段B"]
context="\n\n".join(top_chunks)
def build_prompt_simple(context:str, query:str)->str:
    prompt_template = """请严格依据下面的参考资料回答用户的问题，如果参考资料没有相关信息就如实说明不知道，不要编造内容。
    参考资料：
    {context}
    用户问题：{query}
    回答："""
    
    # query是用户原始提问字符串
    final_prompt = prompt_template.format(context=context, query=query)
    return final_prompt
res = retrieve("什么是RAG", model, chunks, embeds, top_k=2)
print(res)
