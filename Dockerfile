FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# OS 基本ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git \
    build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Deadsnakes PPA で Python 3.12 を導入（distutils は不要）
RUN add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# pip を 3.12 に導入（どちらか片方でOK）
# 1) 標準の ensurepip を使う方法
RUN python3.12 -m ensurepip --upgrade || true

# 2) うまくいかない環境向け（fallback）
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# python/pip のコマンドを 3.12 に向ける
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1

# uv を導入
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# 依存レイヤ（ロックを先にコピーしてキャッシュ最大化）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python 3.12 --no-install-project

# プロジェクト本体
COPY . .
RUN uv sync --frozen --python 3.12

# 起動時の確認
CMD ["uv","run","python","-c","import torch;print('torch',torch.__version__,'CUDA?',torch.cuda.is_available());print('device count',torch.cuda.device_count())"]
