from flask import Flask, request, jsonify, send_from_directory
from flask import Response, stream_with_context
import requests
import json
import os
import time
import chromadb
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")
TAVILY_KEY = os.environ.get("TAVILY_KEY", "")
URL = "https://api.deepseek.com/chat/completions"
HEADERS = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
DATA_DIR = os.environ.get("DATA_DIR", ".")
NOTES_DIR = os.path.join(DATA_DIR, "notes")

HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
MEMORY_META_FILE = os.path.join(DATA_DIR, "memory_meta.json")
CONTEXT_SUMMARY_FILE = os.path.join(DATA_DIR, "context_summary.json")
MAX_HISTORY_PAIRS = 8  # 保留最近 8 轮完整对话，更早的压缩为摘要

BASE_SYSTEM_PROMPT = "你是一个备考助手，帮助用户整理考研知识点。你拥有长期记忆——每次对话后知识点会自动入库。回答用户问题前，先用 search_memory 搜索历史记忆，看看之前是否讨论过相关内容，再结合记忆回答。需要时使用：search_web（搜索网络）、save_note（保存笔记）、read_note（读取笔记）、list_topics（列出笔记）、search_memory（语义搜索记忆）。"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)

os.makedirs(NOTES_DIR, exist_ok=True)
conversation_history = load_history()

# ── 上下文压缩 ─────────────────────────────────────

def load_context_summary():
    if os.path.exists(CONTEXT_SUMMARY_FILE):
        with open(CONTEXT_SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"summary": ""}

def save_context_summary():
    with open(CONTEXT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(context_summary, f, ensure_ascii=False, indent=2)

context_summary = load_context_summary()

def get_system_content():
    """构建带摘要的系统提示词"""
    summary = context_summary.get("summary", "")
    if summary:
        return f"[以下是之前对话中讨论过的知识点摘要，仅供参考]\n{summary}\n\n{BASE_SYSTEM_PROMPT}"
    return BASE_SYSTEM_PROMPT

def _count_pairs(history):
    """统计对话轮次（从索引1开始，跳过system message）"""
    return sum(1 for m in history[1:] if m["role"] == "user")

def summarize_exchange(prev_summary, user_content, assistant_content):
    """调用 DeepSeek 将一轮对话的要点合并进摘要"""
    prompt = f"""将以下新对话要点合并到已有摘要中，输出更新后的摘要（2-5句话，只保留对考研备考有价值的知识点信息）。

已有摘要：{prev_summary if prev_summary else '（尚无摘要）'}

新对话：
用户：{user_content[:300]}
助手：{assistant_content[:500]}

仅输出更新后的摘要，不要额外解释："""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是对话摘要器。输出简洁的要点摘要。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 300
        }, timeout=15)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except:
        return prev_summary

def compress_history():
    """将最早一轮对话压缩进摘要，从历史中移除"""
    global conversation_history, context_summary

    if _count_pairs(conversation_history) <= MAX_HISTORY_PAIRS:
        return

    # 找到第一轮用户消息的位置（跳过 system message）
    first_user_idx = None
    for i in range(1, len(conversation_history)):
        if conversation_history[i]["role"] == "user":
            first_user_idx = i
            break
    if first_user_idx is None:
        return

    # 找到这一轮的用户消息和对应的助手回复（中间可能夹 tool 消息）
    user_msg = conversation_history[first_user_idx]
    assistant_msg = None
    end_idx = first_user_idx + 1
    for i in range(first_user_idx + 1, len(conversation_history)):
        if conversation_history[i]["role"] == "user":
            break
        if conversation_history[i]["role"] == "assistant":
            assistant_msg = conversation_history[i]
        end_idx = i + 1

    if assistant_msg is None:
        return

    # 生成新摘要
    prev = context_summary.get("summary", "")
    new_summary = summarize_exchange(prev, user_msg["content"], assistant_msg["content"])
    context_summary["summary"] = new_summary
    save_context_summary()

    # 从历史中删除这轮消息
    del conversation_history[first_user_idx:end_idx]
    # 更新 system 消息内容
    conversation_history[0]["content"] = get_system_content()
    save_history()

# ── 向量数据库 ──────────────────────────────────────

VECTOR_DIR = os.path.join(DATA_DIR, "vector_db")

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedder

def _norm(vec):
    """L2归一化，保证欧氏距离与余弦相似度一致"""
    mag = sum(x * x for x in vec) ** 0.5
    return [x / mag for x in vec] if mag > 0 else vec

def _embed(text):
    """编码文本并返回归一化向量"""
    return _norm(_get_embedder().encode(text).tolist())

chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
collection = chroma_client.get_or_create_collection(name="study_notes")

# ── 记忆元数据（访问频次、来源、创建时间） ──────────

def load_meta():
    if os.path.exists(MEMORY_META_FILE):
        with open(MEMORY_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_meta(meta):
    with open(MEMORY_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

memory_meta = load_meta()

def _ensure_meta(note_id, topic, source):
    """为新笔记初始化元数据"""
    if note_id not in memory_meta:
        memory_meta[note_id] = {
            "topic": topic,
            "source": source,
            "search_count": 0,
            "created_at": time.time()
        }
        save_meta(memory_meta)

def _record_search(note_ids):
    """记录一次搜索命中，自动为旧笔记补齐元数据"""
    changed = False
    for nid in note_ids:
        if nid not in memory_meta:
            # 从向量库查元数据补齐
            info = collection.get(ids=[nid])
            topic = "未知"
            source = "migrated"
            if info['metadatas'] and info['metadatas'][0]:
                topic = info['metadatas'][0].get('topic', '未知')
                source = info['metadatas'][0].get('source', 'migrated')
            memory_meta[nid] = {"topic": topic, "source": source, "search_count": 0, "created_at": time.time()}
        memory_meta[nid]["search_count"] += 1
        changed = True
    if changed:
        save_meta(memory_meta)

def _importance(note_id, content_len):
    """计算记忆的重要性权重"""
    meta = memory_meta.get(note_id, {})
    # 来源权重
    source_w = {"manual": 1.3, "migrated": 1.1, "auto": 1.0}
    w = source_w.get(meta.get("source", "auto"), 1.0)
    # 访问频次加分：每次搜索命中 +3%，上限 60%
    search_hits = meta.get("search_count", 0)
    w += min(search_hits * 0.03, 0.6)
    # 内容质量加分
    if content_len >= 500:
        w += 0.25
    elif content_len >= 200:
        w += 0.15
    return w

def _check_duplicate(content, threshold=0.75):
    """检查是否已存在高度相似的内容"""
    if collection.count() == 0:
        return False, None
    try:
        q_emb = _embed(content)
        results = collection.query(query_embeddings=[q_emb], n_results=1, include=['embeddings', 'metadatas'])
        ids = results.get('ids', [[]])[0]
        if not ids:
            return False, None
        stored_emb = results['embeddings'][0][0]
        cos = _cosine_sim(q_emb, stored_emb)
        if cos > threshold:
            topic = results['metadatas'][0][0].get('topic', '未知')
            return True, topic
    except:
        pass
    return False, None

def _cosine_sim(a, b):
    """手动计算余弦相似度（兼容 list 和 numpy array）"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


def search_memory(query, n=3):
    """语义搜索笔记，按余弦相似度×重要性权重排序"""
    q_emb = _embed(query)
    # 多取候选，用余弦相似度重排
    fetch_n = max(n * 3, 10)
    candidates = collection.query(
        query_embeddings=[q_emb], n_results=fetch_n, include=['embeddings', 'documents', 'metadatas']
    )
    if not candidates['documents'][0]:
        return "未找到相关笔记"

    scored = []
    for doc, meta, emb, nid in zip(
        candidates['documents'][0],
        candidates['metadatas'][0],
        candidates['embeddings'][0],
        candidates['ids'][0]
    ):
        cos = _cosine_sim(q_emb, emb)
        imp = _importance(nid, len(doc))
        score = cos * imp
        scored.append((score, meta.get('topic', '未知'), doc[:500], nid))

    scored.sort(key=lambda x: x[0], reverse=True)
    _record_search([s[3] for s in scored[:n]])

    output = []
    for score, topic, doc, nid in scored[:n]:
        meta = memory_meta.get(nid, {})
        src_label = {"manual": "手动", "migrated": "迁移", "auto": "自动"}.get(meta.get("source", ""), "")
        hits = meta.get("search_count", 0)
        output.append(f"【{topic}】（得分:{score:.2f} | {src_label}保存 | 被查阅{hits}次）\n{doc[:500]}")
    return ("\n" + "=" * 50 + "\n").join(output)

def save_to_vector(topic, content, source="manual"):
    """将笔记存入向量数据库，记录元数据"""
    note_id = topic.replace(" ", "_")
    emb = _embed(content)
    existing = collection.get(ids=[note_id])
    if existing['ids']:
        collection.update(ids=[note_id], embeddings=[emb], documents=[content], metadatas=[{"topic": topic, "source": source}])
    else:
        collection.add(ids=[note_id], embeddings=[emb], documents=[content], metadatas=[{"topic": topic, "source": source}])
    _ensure_meta(note_id, topic, source)

def migrate_existing_notes():
    """将已有的文件笔记迁移到向量数据库"""
    existing_ids = set(collection.get()['ids'])
    for fname in os.listdir(NOTES_DIR):
        if not fname.endswith(".txt"):
            continue
        topic = fname.replace(".txt", "")
        note_id = topic.replace(" ", "_")
        if note_id in existing_ids:
            continue
        with open(os.path.join(NOTES_DIR, fname), "r", encoding="utf-8") as f:
            content = f.read()
        emb = _embed(content)
        collection.add(ids=[note_id], embeddings=[emb], documents=[content], metadatas=[{"topic": topic, "source": "migrated"}])
        _ensure_meta(note_id, topic, "migrated")

migrate_existing_notes()

# ── 自动记忆 ──────────────────────────────────────

def extract_knowledge(user_msg, assistant_msg):
    """分析对话，决定是否值得记住，并提取知识点摘要"""
    prompt = f"""分析以下对话。如果只是闲聊、问候、与学习无关的话题，返回 worth_saving=false。
如果涉及有价值的学习知识点，提取出知识点并给出主题和1-2句话摘要。

用户消息：{user_msg[:500]}
助手回复：{assistant_msg[:800]}

只返回 JSON（不要其他文字）：
{{"worth_saving": true或false, "notes": [{{"topic": "知识点主题", "content": "摘要内容"}}]}}"""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是知识提取器。严格输出JSON，不要markdown包裹。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }, timeout=15)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text[4:]
        return json.loads(text)
    except:
        return {"worth_saving": False, "notes": []}

def auto_memorize(user_msg, assistant_msg):
    """对话结束后自动提取知识点并写入向量库（带去重）"""
    if len(assistant_msg) < 50:
        return
    result = extract_knowledge(user_msg, assistant_msg)
    if result.get("worth_saving") and result.get("notes"):
        for note in result["notes"]:
            topic = note.get("topic", "").strip()
            content = note.get("content", "").strip()
            if not (topic and content):
                continue
            full_content = f"主题：{topic}\n{content}"
            # 去重检查
            is_dup, existing_topic = _check_duplicate(full_content)
            if is_dup:
                continue
            note_id = f"auto_{int(time.time()*1000)}_{topic.replace(' ', '_')[:30]}"
            emb = _embed(full_content)
            collection.add(
                ids=[note_id],
                embeddings=[emb],
                documents=[full_content],
                metadatas=[{"topic": topic, "source": "auto"}]
            )
            _ensure_meta(note_id, topic, "auto")

# ── 工具函数 ──────────────────────────────────────

def save_note(topic, content):
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    save_to_vector(topic, content)
    return f"笔记已保存：{topic}.txt"

def read_note(topic):
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    if not os.path.exists(path):
        return f"找不到笔记：{topic}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def list_topics():
    files = os.listdir(NOTES_DIR)
    topics = [f.replace(".txt", "") for f in files if f.endswith(".txt")]
    return "已有笔记：" + "、".join(topics) if topics else "还没有任何笔记"

def search_web(query):
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": 3},
            timeout=10
        )
        results = resp.json().get("results", [])
        return "\n\n".join([r["content"] for r in results]) if results else "未找到相关资料"
    except:
        return "搜索失败，请检查网络"

def run_tool(name, args):
    if name == "save_note":    return save_note(**args)
    if name == "read_note":    return read_note(**args)
    if name == "list_topics":  return list_topics()
    if name == "search_web":   return search_web(**args)
    if name == "search_memory":return search_memory(**args)
    return f"未知工具：{name}"

TOOLS = [
    {"type": "function", "function": {
        "name": "save_note",
        "description": "将知识点笔记保存到本地文件",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "笔记主题"},
            "content": {"type": "string", "description": "笔记正文"}
        }, "required": ["topic", "content"]}
    }},
    {"type": "function", "function": {
        "name": "read_note",
        "description": "读取已保存的某个主题笔记",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "要读取的笔记主题"}
        }, "required": ["topic"]}
    }},
    {"type": "function", "function": {
        "name": "list_topics",
        "description": "列出所有已保存的笔记主题",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "搜索网络资料，用于补充知识点",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "语义搜索已保存的笔记，查找与查询相关的内容。当你需要回顾之前学过的知识点时用这个工具，而不是用关键词精确匹配。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要搜索的问题或知识点描述，支持自然语言"},
            "n": {"type": "integer", "description": "返回的结果数量，默认3"}
        }, "required": ["query"]}
    }}
]

# ── 路由 ──────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    events = []  # 记录所有步骤，返回给前端

    messages = [
        {"role": "system", "content": get_system_content()},
        {"role": "user", "content": user_input}
    ]

    while True:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-chat",
            "messages": messages,
            "tools": TOOLS
        })
        msg = resp.json()["choices"][0]["message"]
        messages.append(msg)

        if msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                result = run_tool(name, args)
                # 记录工具调用事件
                events.append({
                    "type": "tool",
                    "name": name,
                    "args": args,
                    "result": result[:200]  # 截断避免太长
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                })
        else:
            events.append({"type": "answer", "content": msg["content"]})
            auto_memorize(user_input, msg["content"])
            break

    return jsonify({"events": events})

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    user_input = request.json.get("message", "")
    conversation_history.append({"role": "user", "content": user_input})

    def generate():
        messages = conversation_history.copy()
        # 注入最新摘要到系统消息
        if messages:
            messages[0]["content"] = get_system_content()
        while True:
            resp = requests.post(URL, headers=HEADERS, json={
                "model": "deepseek-chat",
                "messages": messages,
                "tools": TOOLS
            })
            msg = resp.json()["choices"][0]["message"]
            messages.append(msg)

            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    result = run_tool(name, args)
                    yield f"data: {json.dumps({'type':'tool','name':name,'result':result[:100]})}\n\n"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
            else:
                stream_resp = requests.post(URL, headers=HEADERS, json={
                    "model": "deepseek-chat",
                    "messages": messages[:-1],
                    "tools": TOOLS,
                    "stream": True
                }, stream=True)

                full_content = ""
                for line in stream_resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token:
                                full_content += token
                                yield f"data: {json.dumps({'type':'text','char':token})}\n\n"
                        except:
                            pass

                yield f"data: {json.dumps({'type':'done'})}\n\n"
                conversation_history.append({"role": "assistant", "content": full_content})
                save_history()
                compress_history()
                auto_memorize(user_input, full_content)
                break

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 备考助手启动：http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
