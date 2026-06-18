# PodcastZ 后端镜像 —— CPU 转写 + ffmpeg + Python
# GPU 版见 docker-compose.yml 注释

FROM python:3.11-slim AS base

# 系统依赖(ffmpeg 作为后备;主用 imageio-ffmpeg 打包二进制)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖(利用 Docker 层缓存)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
        "pydantic>=2.6" "pydantic-settings>=2.2" \
        "pydub>=0.25.1" "ffmpeg-python>=0.2.0" "imageio-ffmpeg>=0.4.9" \
        "numpy>=1.26" "soundfile>=0.12.1" "scipy>=1.11" \
        "faster-whisper>=1.0.3" "ctranslate2>=4.1" \
        "jieba>=0.42.1" "openai>=1.30" "httpx>=0.27" \
        "loguru>=0.7.2" "tqdm>=4.66"

# 拷贝源码
COPY backend ./backend
COPY frontend ./frontend

# 数据卷
VOLUME ["/app/data", "/app/models"]
ENV MODELS_DIR=/app/models

EXPOSE 8000

# 预下载模型可选(用环境变量);否则首次请求时下载
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
