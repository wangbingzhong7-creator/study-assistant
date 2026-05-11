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

EXPOSE ${PORT:-5000}

# gunicorn 生产模式：1 worker（嵌入模型较重，多worker OOM）
# PORT 由 Railway 自动注入，本地默认 5000
CMD exec gunicorn --bind "0.0.0.0:${PORT:-5000}" --workers 1 --timeout 120 app_fj:app
