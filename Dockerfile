# Dockerfile
FROM python:3.11-slim

# 기본 환경
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

RUN  apt-get update && apt-get install -y --no-install-recommends curl fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# 필요한 패키지
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

# 소스 복사
COPY . .

# 기본 런타임 환경(필요 시 .env에서 덮어씀)
ENV TORCH_DTYPE=bfloat16


# 헬스체크가 찍는 access log 줄이기(--no-access-log)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
