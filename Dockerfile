FROM python:3.11-slim

WORKDIR /app

# 系统依赖 (chromadb 需要 gcc 等编译依赖)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 数据持久化目录
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

# 嵌入模型首次下载预热（构建时缓存）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

EXPOSE 5000

# gunicorn 生产模式：1 worker（因为嵌入模型较重，多 worker 会 OOM）
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--preload", "app_fj:app"]
