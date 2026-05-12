from flask import Flask, request, jsonify, send_from_directory
from flask import Response, stream_with_context
import requests
import json
import os
import time
import smtplib
from email.message import EmailMessage
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")
TAVILY_KEY = os.environ.get("TAVILY_KEY", "")
# 反馈邮箱配置（QQ邮箱SMTP）
FEEDBACK_EMAIL = os.environ.get("FEEDBACK_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
AVATAR_URL = os.environ.get("AVATAR_URL", "")
URL = "https://api.deepseek.com/chat/completions"
HEADERS = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
DATA_DIR = os.environ.get("DATA_DIR", ".")
NOTES_DIR = os.path.join(DATA_DIR, "notes")

HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
MEMORY_META_FILE = os.path.join(DATA_DIR, "memory_meta.json")
CONTEXT_SUMMARY_FILE = os.path.join(DATA_DIR, "context_summary.json")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
SESSIONS_META_FILE = os.path.join(DATA_DIR, "sessions.json")
MAX_HISTORY_PAIRS = 8  # 保留最近 8 轮完整对话，更早的压缩为摘要

SUBJECTS = ["政治", "英语", "数学", "专业课", "其他"]

BASE_SYSTEM_PROMPT = "你是一个备考助手，帮助用户整理考研知识点。你拥有长期记忆——每次对话后知识点会自动入库。回答用户问题前，先用 search_memory 搜索历史记忆，看看之前是否讨论过相关内容，再结合记忆回答。保存笔记时请根据内容判断所属科目（政治/英语/数学/专业课/其他），搜索记忆时可用科目筛选。笔记保存在服务器，用户可随时查看/搜索/删除。保存后告诉用户可以通过下载链接把文件保存到本地（如：https://你的域名/download/主题名），不要道歉说没法保存。需要时使用：search_web（搜索网络）、save_note（保存笔记）、read_note（读取笔记）、list_topics（列出笔记）、search_memory（语义搜索记忆）、generate_quiz（根据已学知识点出题）、delete_note（删除笔记）、update_note（修改笔记内容）。"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

def save_history():
    sid = get_current_session_id()
    if sid:
        _save_session_history(sid, conversation_history)
        # 更新会话预览（最后一条助手消息的前30字）
        for s in sessions_meta["list"]:
            if s["id"] == sid:
                for m in reversed(conversation_history):
                    if m["role"] == "assistant":
                        s["preview"] = m["content"][:30]
                        s["title"] = s.get("title", "默认会话")
                        break
                _save_sessions_meta()
                break
    else:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)

os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ── 多会话管理 ─────────────────────────────────────

def _load_sessions_meta():
    if os.path.exists(SESSIONS_META_FILE):
        with open(SESSIONS_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current": "", "list": []}

def _save_sessions_meta():
    with open(SESSIONS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions_meta, f, ensure_ascii=False, indent=2)

sessions_meta = _load_sessions_meta()

def migrate_to_sessions():
    """首次启动：把旧的 history.json 迁移为默认会话"""
    global sessions_meta
    if sessions_meta.get("list"):
        return  # 已有会话，跳过迁移
    sid = f"session_{int(time.time())}"
    session = {
        "id": sid,
        "title": "默认会话",
        "created_at": int(time.time()),
        "preview": ""
    }
    sessions_meta["list"].append(session)
    sessions_meta["current"] = sid
    # 迁移旧历史
    old_history = load_history()
    _save_session_history(sid, old_history)
    _save_sessions_meta()
    # 删除旧 history.json
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def _get_session_path(sid):
    return os.path.join(SESSIONS_DIR, f"{sid}.json")

def _load_session_history(sid):
    path = _get_session_path(sid)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

def _save_session_history(sid, history):
    with open(_get_session_path(sid), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_current_history():
    """返回当前会话的对话历史"""
    sid = sessions_meta.get("current", "")
    if sid:
        return _load_session_history(sid)
    # fallback: 用旧的全局变量
    return load_history()

def get_current_session_id():
    return sessions_meta.get("current", "")

migrate_to_sessions()
conversation_history = get_current_history()

def _auto_name_session(user_msg):
    """用第一条用户消息给会话命名"""
    sid = get_current_session_id()
    if not sid:
        return
    for s in sessions_meta["list"]:
        if s["id"] == sid and s["title"] in ("新会话", "默认会话"):
            s["title"] = user_msg[:15]
            _save_sessions_meta()
            break

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

# 使用 ChromaDB 内置 ONNX 嵌入（免 PyTorch，内存仅 ~50MB）
_ef = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)
collection = chroma_client.get_or_create_collection(name="study_notes", embedding_function=_ef)

def _sim_from_dist(dist):
    """ChromaDB ONNX 返回余弦距离，转成相似度 [0,1]"""
    return max(0, 1 - dist)

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

def _ensure_meta(note_id, topic, source, subject=""):
    """为新笔记初始化元数据"""
    if note_id not in memory_meta:
        memory_meta[note_id] = {
            "topic": topic,
            "source": source,
            "subject": subject,
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
        results = collection.query(query_texts=[content], n_results=1, include=['metadatas', 'distances'])
        ids = results.get('ids', [[]])[0]
        if not ids:
            return False, None
        dist = results['distances'][0][0]
        if _sim_from_dist(dist) > threshold:
            topic = results['metadatas'][0][0].get('topic', '未知')
            return True, topic
    except:
        pass
    return False, None


def search_memory(query, n=3, subject=""):
    """语义搜索笔记，按相似度×重要性权重排序，可按科目过滤"""
    fetch_n = max(n * 3, 10)
    candidates = collection.query(
        query_texts=[query], n_results=max(fetch_n, collection.count()),
        include=['distances', 'documents', 'metadatas']
    )
    if not candidates['documents'][0]:
        return "未找到相关笔记"

    scored = []
    for doc, meta, dist, nid in zip(
        candidates['documents'][0],
        candidates['metadatas'][0],
        candidates['distances'][0],
        candidates['ids'][0]
    ):
        doc_subject = meta.get('subject', '') or memory_meta.get(nid, {}).get('subject', '')
        if subject and doc_subject and doc_subject != subject:
            continue
        sim = _sim_from_dist(dist)
        imp = _importance(nid, len(doc))
        score = sim * imp
        scored.append((score, meta.get('topic', '未知'), doc[:500], nid, doc_subject))

    if not scored:
        subj_hint = f"在科目「{subject}」中" if subject else ""
        return f"{subj_hint}未找到相关笔记"

    scored.sort(key=lambda x: x[0], reverse=True)
    _record_search([s[3] for s in scored[:n]])

    output = []
    for score, topic, doc, nid, doc_subj in scored[:n]:
        meta = memory_meta.get(nid, {})
        src_label = {"manual": "手动", "migrated": "迁移", "auto": "自动"}.get(meta.get("source", ""), "")
        hits = meta.get("search_count", 0)
        subj_tag = f" | {doc_subj}" if doc_subj else ""
        output.append(f"【{topic}】（得分:{score:.2f}{subj_tag} | {src_label}保存 | 被查阅{hits}次）\n{doc[:500]}")
    return ("\n" + "=" * 50 + "\n").join(output)

def save_to_vector(topic, content, source="manual", subject=""):
    """将笔记存入向量数据库，记录元数据（含科目分类）"""
    note_id = topic.replace(" ", "_")
    existing = collection.get(ids=[note_id])
    if existing['ids']:
        collection.update(ids=[note_id], documents=[content],
                          metadatas=[{"topic": topic, "source": source, "subject": subject}])
    else:
        collection.add(ids=[note_id], documents=[content],
                       metadatas=[{"topic": topic, "source": source, "subject": subject}])
    _ensure_meta(note_id, topic, source, subject)

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
        collection.add(ids=[note_id], documents=[content], metadatas=[{"topic": topic, "source": "migrated"}])
        _ensure_meta(note_id, topic, "migrated")

migrate_existing_notes()

# ── 自动记忆 ──────────────────────────────────────

def extract_knowledge(user_msg, assistant_msg):
    """分析对话，决定是否值得记住，并提取知识点摘要及所属科目"""
    prompt = f"""分析以下对话。如果只是闲聊、问候、与学习无关的话题，返回 worth_saving=false。
如果涉及有价值的学习知识点，提取出知识点，给出主题、内容摘要、所属科目。

科目从以下选择：政治、英语、数学、专业课、其他

用户消息：{user_msg[:500]}
助手回复：{assistant_msg[:800]}

只返回 JSON（不要其他文字）：
{{"worth_saving": true或false, "notes": [{{"topic": "知识点主题", "content": "摘要内容", "subject": "科目"}}]}}"""

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
            subject = note.get("subject", "").strip()
            if not (topic and content):
                continue
            full_content = f"主题：{topic}\n{content}"
            # 去重检查
            is_dup, existing_topic = _check_duplicate(full_content)
            if is_dup:
                continue
            note_id = f"auto_{int(time.time()*1000)}_{topic.replace(' ', '_')[:30]}"
            collection.add(
                ids=[note_id],
                documents=[full_content],
                metadatas=[{"topic": topic, "source": "auto", "subject": subject}]
            )
            _ensure_meta(note_id, topic, "auto", subject)

# ── 工具函数 ──────────────────────────────────────

def save_note(topic, content, subject=""):
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    save_to_vector(topic, content, subject=subject)
    subject_tag = f"[{subject}] " if subject else ""
    return f"笔记已保存：{subject_tag}{topic}.txt（下载链接：/download/{topic}）"

def read_note(topic):
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    if not os.path.exists(path):
        return f"找不到笔记：{topic}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def delete_note(topic):
    """删除指定主题的笔记（文件和向量库）"""
    # 精确匹配笔记ID
    note_id = topic.replace(" ", "_")
    all_ids = collection.get()['ids']

    # 尝试精确匹配
    matched = [nid for nid in all_ids if nid == note_id]
    # 不区分大小写的模糊匹配（针对自动生成的笔记）
    if not matched:
        matched = [nid for nid in all_ids if topic in nid or nid in note_id]

    if not matched:
        return f"找不到笔记「{topic}」，可能已被删除或主题名不准确。请用 list_topics 查看已有笔记。"

    for nid in matched:
        collection.delete(ids=[nid])
        if nid in memory_meta:
            del memory_meta[nid]
        save_meta(memory_meta)

    # 删除对应的文件
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    if os.path.exists(path):
        os.remove(path)

    return f"已删除笔记「{topic}」（{len(matched)} 条记录）"

def update_note(topic, content, subject=""):
    """更新已有笔记的内容，找不到则创建新笔记"""
    note_id = topic.replace(" ", "_")
    all_ids = collection.get()['ids']

    # 先删旧数据再写入
    matched = [nid for nid in all_ids if nid == note_id]
    if not matched:
        matched = [nid for nid in all_ids if topic in nid]
    for nid in matched:
        collection.delete(ids=[nid])
        if nid in memory_meta:
            del memory_meta[nid]

    # 写入新内容
    save_to_vector(topic, content, subject=subject)
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    subject_tag = f"[{subject}] " if subject else ""
    action = "更新" if matched else "创建"
    return f"笔记已{action}：{subject_tag}{topic}.txt"

def list_topics(subject=""):
    """列出已有笔记，可按科目筛选，返回按科目分组的结果"""
    # 从向量库和内存元数据汇总
    subjects = {}
    all_ids = collection.get()['ids']
    for nid in all_ids:
        meta = memory_meta.get(nid, {})
        s = meta.get("subject", "其他") or "其他"
        t = meta.get("topic", nid)
        if subject and s != subject:
            continue
        subjects.setdefault(s, []).append(t)

    if not subjects:
        return "还没有任何笔记"

    lines = []
    for s in SUBJECTS:
        if s in subjects:
            topics = subjects.pop(s)
            lines.append(f"【{s}】{'、'.join(topics)}")
    # 剩余不在预设科目里的
    for s, topics in subjects.items():
        lines.append(f"【{s}】{'、'.join(topics)}")
    return "\n".join(lines)

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

def generate_quiz(subject="", count=5, qtype="mixed"):
    """根据向量库中的知识点生成练习题"""
    # 1. 从向量库提取素材
    if subject:
        topics_str = list_topics(subject=subject)
        knowledge = search_memory(f"{subject} 核心概念 要点 考点", n=8, subject=subject)
        # 科目过滤没结果时回退到全库搜索
        if "未找到" in knowledge:
            knowledge = search_memory(f"{subject} 核心概念", n=5)
    else:
        topics_str = list_topics()
        knowledge = search_memory("核心概念 要点 考点", n=8)

    if "未找到" in knowledge and "还没有" in topics_str:
        return "还没有任何笔记，无法出题。请先保存一些知识点。"

    # 2. 让 LLM 根据知识素材出题
    prompt = f"""根据以下知识点素材，生成{count}道练习题。必须基于素材内容出题，不能编造素材中没有的知识点。

知识点素材：
{knowledge[:2500]}

已学主题：{topics_str[:500]}

题目类型：{qtype}（mixed=混合题型，choice=选择题，qa=简答题，fill=填空题）
科目：{subject if subject else '不限'}

要求：
- 每道题标注题号、题型和分值
- 选择题4个选项，只有一个正确答案
- 简答题要求2-4句话回答
- 出题后附上正确答案和简要解析
- 题目难度适中，覆盖核心知识点
- 如果素材不足，减少题目数量

直接输出题目："""

    try:
        resp = requests.post(URL, headers=HEADERS, json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是考研出题专家，出题精准、解析清晰。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }, timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except:
        return "出题失败，请稍后重试"

def run_tool(name, args):
    if name == "save_note":    return save_note(topic=args.get("topic",""), content=args.get("content",""), subject=args.get("subject",""))
    if name == "read_note":    return read_note(**args)
    if name == "list_topics":  return list_topics(subject=args.get("subject",""))
    if name == "search_web":   return search_web(**args)
    if name == "search_memory":return search_memory(query=args.get("query",""), n=args.get("n",3), subject=args.get("subject",""))
    if name == "generate_quiz":return generate_quiz(subject=args.get("subject",""), count=args.get("count",5), qtype=args.get("qtype","mixed"))
    if name == "delete_note":  return delete_note(topic=args.get("topic",""))
    if name == "update_note":  return update_note(topic=args.get("topic",""), content=args.get("content",""), subject=args.get("subject",""))
    return f"未知工具：{name}"

TOOLS = [
    {"type": "function", "function": {
        "name": "save_note",
        "description": "将知识点笔记保存到本地文件。需要根据内容判断所属科目。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "笔记主题"},
            "content": {"type": "string", "description": "笔记正文"},
            "subject": {"type": "string", "description": "科目：政治/英语/数学/专业课/其他"}
        }, "required": ["topic", "content", "subject"]}
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
        "description": "列出所有已保存的笔记主题，可按科目筛选",
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string", "description": "可选，按科目筛选：政治/英语/数学/专业课/其他"}
        }}
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
        "description": "语义搜索已保存的笔记，可按科目过滤。当你需要回顾之前学过的知识点时用这个工具。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要搜索的问题或知识点描述，支持自然语言"},
            "n": {"type": "integer", "description": "返回的结果数量，默认3"},
            "subject": {"type": "string", "description": "可选，按科目过滤：政治/英语/数学/专业课/其他"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "generate_quiz",
        "description": "根据已保存的知识点生成练习题。用户要求做题/出题/测试时调用。自动从向量库提取知识素材生成题目，附正确答案和解析。",
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string", "description": "出题科目：政治/英语/数学/专业课/其他，不填则综合出题"},
            "count": {"type": "integer", "description": "题目数量，默认5"},
            "qtype": {"type": "string", "description": "题型：mixed(混合)/choice(选择)/qa(简答)/fill(填空)，默认mixed"}
        }}
    }},
    {"type": "function", "function": {
        "name": "delete_note",
        "description": "删除指定主题的笔记，同时从文件和向量库中移除。用户说删除/去掉某条笔记时调用。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "要删除的笔记主题名"}
        }, "required": ["topic"]}
    }},
    {"type": "function", "function": {
        "name": "update_note",
        "description": "修改或更新已有笔记的内容。用户说修改/更正/补充某条笔记时调用。找不到旧笔记则自动创建。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "要修改的笔记主题名"},
            "content": {"type": "string", "description": "新的笔记正文"},
            "subject": {"type": "string", "description": "科目：政治/英语/数学/专业课/其他"}
        }, "required": ["topic", "content"]}
    }}
]

# ── 学习数据面板 ────────────────────────────────────

def get_stats():
    """聚合学习统计数据"""
    meta_list = list(memory_meta.values())
    total = len(meta_list)

    # 科目分布
    subjects = {}
    for m in meta_list:
        s = m.get("subject", "其他") or "其他"
        subjects[s] = subjects.get(s, 0) + 1

    # 来源分布
    manual = sum(1 for m in meta_list if m.get("source") == "manual")
    auto = sum(1 for m in meta_list if m.get("source") == "auto")

    # 最多被查阅的
    most = sorted(meta_list, key=lambda m: m.get("search_count", 0), reverse=True)[:5]
    top_notes = [{"topic": m["topic"], "hits": m.get("search_count", 0), "subject": m.get("subject", "")}
                 for m in most if m.get("search_count", 0) > 0]

    # 最近添加的
    recent = sorted(meta_list, key=lambda m: m.get("created_at", 0), reverse=True)[:5]
    recent_notes = [{"topic": m["topic"], "subject": m.get("subject", ""),
                     "date": time.strftime("%m-%d", time.localtime(m.get("created_at", 0)))}
                    for m in recent]

    # 总搜索次数
    total_searches = sum(m.get("search_count", 0) for m in meta_list)

    return {
        "total_notes": total,
        "manual_notes": manual,
        "auto_notes": auto,
        "subjects": subjects,
        "top_notes": top_notes,
        "recent_notes": recent_notes,
        "total_searches": total_searches
    }

# ── 路由 ──────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/config")
def site_config():
    return jsonify({"avatar_url": AVATAR_URL})

@app.route("/download/<topic>")
def download_note(topic):
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    if not os.path.exists(path):
        return "笔记不存在", 404
    return send_from_directory(NOTES_DIR, f"{topic}.txt", as_attachment=True, download_name=f"{topic}.txt")

@app.route("/stats")
def stats():
    return jsonify(get_stats())

@app.route("/sessions", methods=["GET", "POST", "DELETE"])
def sessions_api():
    global conversation_history, sessions_meta

    if request.method == "GET":
        # 列表
        return jsonify({
            "current": sessions_meta.get("current", ""),
            "list": sessions_meta.get("list", [])
        })

    elif request.method == "POST":
        # 创建新会话
        data = request.get_json() or {}
        title = data.get("title", "").strip() or "新会话"
        sid = f"session_{int(time.time()*1000)}"
        sess = {
            "id": sid,
            "title": title,
            "created_at": int(time.time()),
            "preview": ""
        }
        sessions_meta["list"].append(sess)
        sessions_meta["current"] = sid
        _save_session_history(sid, [{"role": "system", "content": BASE_SYSTEM_PROMPT}])
        _save_sessions_meta()
        conversation_history = _load_session_history(sid)
        return jsonify({"status": "ok", "session": sess})

    elif request.method == "DELETE":
        # 删除会话
        sid = request.args.get("id", "")
        if not sid or len(sessions_meta["list"]) <= 1:
            return jsonify({"status": "error", "msg": "至少保留一个会话"}), 400
        sessions_meta["list"] = [s for s in sessions_meta["list"] if s["id"] != sid]
        # 删文件
        path = _get_session_path(sid)
        if os.path.exists(path):
            os.remove(path)
        # 如果删的是当前会话，切到第一个
        if sessions_meta.get("current") == sid:
            sessions_meta["current"] = sessions_meta["list"][0]["id"]
            conversation_history = _load_session_history(sessions_meta["current"])
        _save_sessions_meta()
        return jsonify({"status": "ok"})

@app.route("/sessions/<sid>/switch", methods=["POST"])
def switch_session(sid):
    global conversation_history, sessions_meta
    # 验证 sid 存在
    valid = any(s["id"] == sid for s in sessions_meta.get("list", []))
    if not valid:
        return jsonify({"status": "error", "msg": "会话不存在"}), 404
    sessions_meta["current"] = sid
    _save_sessions_meta()
    conversation_history = _load_session_history(sid)
    return jsonify({"status": "ok"})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    msg_text = data.get("message", "").strip()
    if not msg_text or len(msg_text) < 2:
        return jsonify({"status": "error", "msg": "内容太短"}), 400

    # 保存到本地文件作为备份
    feedback_file = os.path.join(DATA_DIR, "feedback.json")
    feedbacks = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            feedbacks = json.load(f)
    feedbacks.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": msg_text
    })
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, ensure_ascii=False, indent=2)

    # 如果配置了邮箱，发送邮件通知
    email_ok = False
    if FEEDBACK_EMAIL and SMTP_PASSWORD:
        mail = EmailMessage()
        mail["Subject"] = f"[备考助手反馈] {msg_text[:30]}..."
        mail["From"] = FEEDBACK_EMAIL
        mail["To"] = FEEDBACK_EMAIL
        mail.set_content(f"收到一条用户反馈：\n\n{msg_text}\n\n时间：{feedbacks[-1]['time']}")
        # 先试 587 + STARTTLS，再试 465 SSL
        for port, use_ssl in [(587, False), (465, True)]:
            try:
                if use_ssl:
                    server = smtplib.SMTP_SSL("smtp.qq.com", port, timeout=10)
                else:
                    server = smtplib.SMTP("smtp.qq.com", port, timeout=10)
                    server.starttls()
                server.login(FEEDBACK_EMAIL, SMTP_PASSWORD)
                server.send_message(mail)
                server.quit()
                email_ok = True
                break
            except:
                continue

    return jsonify({"status": "ok", "email_sent": email_ok})

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
    _auto_name_session(user_input)
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
