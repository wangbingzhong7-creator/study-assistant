NEW = """  <style>
    :root {
      --bg:       #1b1c20;
      --surface:  #25262e;
      --border:   #353740;
      --accent:   #d4a574;
      --accent2:  #b8956a;
      --gold:     #e0b868;
      --green:    #6fb38a;
      --red:      #d97474;
      --text:     #ddd8cf;
      --muted:    #8a857c;
      --mono:     'JetBrains Mono', monospace;
      --serif:    'Noto Serif SC', serif;
      --shadow:   0 2px 8px rgba(0,0,0,0.15);
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--serif);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
    }

    .logo {
      width: 34px; height: 34px;
      background: linear-gradient(135deg, var(--accent), #e8c98e);
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 17px;
      box-shadow: 0 2px 6px rgba(212,165,116,0.2);
    }

    header h1 {
      font-size: 17px;
      font-weight: 600;
      color: var(--text);
      letter-spacing: 1px;
    }

    .subtitle {
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.5px;
      margin-left: auto;
      opacity: 0.6;
    }

    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      scroll-behavior: smooth;
    }

    #chat::-webkit-scrollbar { width: 3px; }
    #chat::-webkit-scrollbar-track { background: transparent; }
    #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    .welcome {
      text-align: center;
      padding: 48px 20px;
      animation: fadeUp 0.5s ease;
    }

    .welcome .icon { font-size: 56px; margin-bottom: 18px; }

    .welcome h2 {
      font-size: 21px;
      font-weight: 600;
      margin-bottom: 10px;
      color: var(--text);
    }

    .welcome p {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.8;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: center;
      margin-top: 24px;
    }

    .chip {
      padding: 10px 18px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 24px;
      font-size: 13px;
      color: var(--text);
      cursor: pointer;
      transition: all 0.2s ease;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .chip:hover {
      border-color: var(--accent);
      color: var(--accent);
      transform: translateY(-1px);
      box-shadow: var(--shadow);
    }

    .msg {
      display: flex;
      gap: 10px;
      animation: fadeUp 0.3s ease;
    }

    .msg.user { flex-direction: row-reverse; }

    .avatar {
      width: 30px; height: 30px;
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px;
      flex-shrink: 0;
    }

    .avatar.ai {
      background: linear-gradient(135deg, #d4a574, #c4905a);
      box-shadow: 0 2px 4px rgba(212,165,116,0.15);
    }

    .avatar.user-av {
      background: var(--surface);
      border: 1px solid var(--border);
    }

    .bubble {
      max-width: 75%;
      padding: 14px 18px;
      border-radius: 16px;
      font-size: 14px;
      line-height: 1.75;
    }

    .msg.user .bubble {
      background: #302c26;
      border: 1px solid #403a30;
      border-top-right-radius: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    .msg.ai .bubble {
      background: var(--surface);
      border: 1px solid var(--border);
      border-top-left-radius: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    .answer-text.typing::after {
      content: '|';
      animation: blink 1s step-end infinite;
      color: var(--accent);
      font-weight: 300;
    }
    @keyframes blink { 50% { opacity: 0; } }

    .tool-card {
      margin: 8px 0;
      padding: 10px 14px;
      background: rgba(212,165,116,0.04);
      border: 1px solid rgba(212,165,116,0.15);
      border-radius: 10px;
      font-size: 12px;
    }

    .tool-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }

    .tool-badge {
      padding: 2px 8px;
      background: rgba(212,165,116,0.12);
      border: 1px solid rgba(212,165,116,0.2);
      border-radius: 5px;
      color: var(--accent);
      font-size: 10px;
    }

    .tool-name { color: var(--accent); font-weight: 500; }
    .tool-args { color: var(--muted); font-size: 11px; }

    .tool-result {
      margin-top: 6px;
      padding-top: 6px;
      border-top: 1px solid var(--border);
      color: var(--green);
      font-size: 11px;
      opacity: 0.85;
    }

    footer {
      padding: 14px 16px 18px;
      border-top: 1px solid var(--border);
      background: var(--bg);
      flex-shrink: 0;
    }

    .input-wrap {
      display: flex;
      gap: 10px;
      align-items: flex-end;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px 16px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }

    .input-wrap:focus-within {
      border-color: rgba(212,165,116,0.4);
      box-shadow: 0 0 0 3px rgba(212,165,116,0.06);
    }

    #input {
      flex: 1;
      background: none;
      border: none;
      outline: none;
      color: var(--text);
      font-family: var(--serif);
      font-size: 14px;
      resize: none;
      line-height: 1.6;
      max-height: 120px;
      overflow-y: auto;
    }

    #input::placeholder { color: var(--muted); opacity: 0.6; }

    #send {
      width: 34px; height: 34px;
      background: linear-gradient(135deg, var(--accent), #c4905a);
      border: none;
      border-radius: 12px;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      color: white;
      font-size: 15px;
      flex-shrink: 0;
      transition: all 0.2s ease;
      box-shadow: 0 2px 6px rgba(212,165,116,0.2);
    }

    #send:hover {
      transform: scale(1.04);
      box-shadow: 0 3px 10px rgba(212,165,116,0.3);
    }
    #send:disabled { opacity: 0.3; transform: none; cursor: not-allowed; box-shadow: none; }

    .thinking {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      border-top-left-radius: 6px;
      font-size: 13px;
      color: var(--muted);
      animation: fadeUp 0.25s ease;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    .dots { display: flex; gap: 5px; }
    .dot {
      width: 6px; height: 6px;
      background: var(--accent);
      border-radius: 50%;
      animation: bounce 1.3s infinite;
      opacity: 0.5;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-4px); opacity: 1; }
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .bubble strong { color: var(--gold); font-weight: 600; }
    .bubble em { color: var(--accent); font-style: normal; }
    .bubble code {
      background: rgba(255,255,255,0.05);
      padding: 1px 6px;
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--green);
    }

    .hint {
      text-align: center;
      font-size: 11px;
      color: var(--muted);
      margin-top: 8px;
      opacity: 0.5;
    }"""

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

s1 = content.find('  <style>')
e1 = content.find('  </style>', s1) + len('  </style>')
content = content[:s1] + NEW + content[e1:]

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('CSS replaced, new length:', len(content))
# Verify
assert content.count('<style>') == 1
assert content.count('</style>') == 1
assert '#1b1c20' in content
assert 'answer-text.typing::after' in content
print('Verification OK')
