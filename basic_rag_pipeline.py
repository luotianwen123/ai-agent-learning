from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import tiktoken
import requests

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
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
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

def calc_available_chunk_quota(model_max_window: int,
                               system_prompt: str,
                               user_query: str,
                               tokenizer,
                               reserve_output_token: int):
    """
    计算还能分给检索上下文chunk的token配额
    返回：可用token额度，小于等于0代表没有余量放知识库内容
    :param model_max_window: 模型上下文窗口大小
    :param system_prompt: 系统提示词
    :param user_query: 用户提问
    :param tokenizer: tiktoken分词器实例
    :param reserve_output_token: 预先留给大模型回答输出的token额度
    """
    sys_tokens= len(tokenizer.encode(system_prompt))
    query_tokens = len(tokenizer.encode(user_query))
    available_chunk_token = model_max_window - sys_tokens - query_tokens - reserve_output_token
    if available_chunk_token <= 0:
        return 0
    return available_chunk_token

def clip_context_by_max_token(chunk_list, token_limit, tokenizer):
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
    """
    组装RAG完整prompt
    safe_context: 已经经过token裁剪的知识库文本，可能为空字符串
    """
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
    max_output_tokens:int = 512
) -> str:
    url = "填入你的接口地址"
    headers = {
        "Authorization": f"Bearer {bearer_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role":"user", "content": prompt_text}
        ],
        "max_tokens": max_output_tokens
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if resp.status_code == 429:
            raise RuntimeError("接口429限流：请求过于频繁")
        if resp.status_code != 200:
            raise RuntimeError(f"接口请求失败，status_code:{resp.status_code}, {resp.text}")

        resp_json = resp.json()
        result_text = resp_json["choices"][0]["message"]["content"]
        return result_text

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络请求异常：{str(e)}")

demo_doc = """Agent（智能体）可以自主规划任务，调用工具，读取记忆。
RAG检索增强生成，通过知识库检索，给大模型补充外部资料，减少幻觉。
文本分块是RAG第一步，合理的分块大小直接影响检索效果。分块过大混入无关信息；分块过小丢失完整语义。"""

if __name__ == "__main__":
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    chunks = recursive_split(demo_doc, max_chunk_size=150, overlap=20)

    vector_store: list[ChunkItem] = []
    for c in chunks:
        emb = model.encode(c).tolist()
        vector_store.append(ChunkItem(text=c, vector=emb))

    query = "什么是RAG？"
    retrieved_chunks = retrieve(query, model, vector_store, top_k=2)

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
        chunk_list=retrieved_chunks,
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

    # =========对接大模型接口（此处需要替换为真实信息）=========
    BEARER_KEY = "在此填入你的key"
    LLM_MODEL_NAME = "在此填入模型名称"

    try:
        answer = llm_chat(bearer_key=BEARER_KEY, model_name=LLM_MODEL_NAME, prompt_text=final_prompt)
        print("\n====大模型返回回答====")
        print(answer)
    except RuntimeError as e:
        print(f"【接口调用异常】{e}")
