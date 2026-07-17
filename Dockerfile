# 馬達即時診斷系統 WebUI
# multi-arch：linux/amd64 與 linux/arm64 皆可建置
#   docker buildx build --platform linux/arm64,linux/amd64 -t motor-diagnosis .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch 用 CPU 專用 index（amd64 取 +cpu 精簡版；arm64 自動回退 PyPI 的 aarch64 wheel）
COPY requirements-web.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-web.txt

COPY app ./app
COPY config.yaml best_motor_cnn_weights.pth ./

# 執行期狀態目錄（compose 掛載 volume）
RUN mkdir -p /app/saved_data /app/state

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
