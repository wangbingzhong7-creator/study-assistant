from flask import Flask, request, jsonify, send_from_directory
from flask import Response, stream_with_context
import requests
import json
import os

app = Flask(__name__)

API_KEY = "sk-940a51af84d0425989d4043c33a06a16"
URL = "https://api.deepseek.com/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
NOTES_DIR = "notes"
TAVILY_KEY = "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"

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是一个备考助手，帮助用户整理考研知识点。需要时主动使用工具搜索、保存、读取笔记。"}]

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)

os.makedirs(NOTES_DIR, exist_ok=True)
conversation_history = load_history()

# ── 工具函数 ──────────────────────────────────────

def save_note(topic, content):
    path = os.path.join(NOTES_DIR, f"{topic}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
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
    if name == "save_note":   return save_note(**args)
    if name == "read_note":   return read_note(**args)
    if name == "list_topics": return list_topics()
    if name == "search_web":  return search_web(**args)
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
    }}
]

# ── 路由 ──────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    events = []  # 记录所有步骤，返回给前端

    messages = [
        {"role": "system", "content": "你是一个备考助手，帮助用户整理考研知识点。需要时主动使用工具搜索、保存、读取笔记。"},
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
            break

    return jsonify({"events": events})

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    user_input = request.json.get("message", "")
    conversation_history.append({"role": "user", "content": user_input})

    def generate():
        messages = conversation_history.copy()
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
                break

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    print("🚀 备考助手启动：http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
