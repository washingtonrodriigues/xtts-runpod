FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# dependências
RUN pip install --upgrade pip
RUN pip install \
    runpod \
    TTS \
    torch==2.2.2+cu118 \
    torchaudio==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# baixa o modelo no build (MUITO IMPORTANTE)
RUN python - <<EOF
from TTS.api import TTS
TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
EOF

COPY handler.py .

CMD ["python", "handler.py"]
