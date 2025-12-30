FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 🔑 Licença Coqui (TEM que estar no build)
ENV COQUI_TOS_AGREED=1
ENV COQUI_COMMERCIAL_LICENSE=0

# performance
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    sox \
    espeak-ng \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install \
    runpod \
    TTS \
    torch==2.2.2+cu118 \
    torchaudio==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 🔥 Download do modelo NO BUILD (CPU)
RUN python - <<EOF
import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["COQUI_COMMERCIAL_LICENSE"] = "0"

from TTS.api import TTS
print("Baixando XTTS v2...")
TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)
print("Download concluído")
EOF

COPY handler.py .

CMD ["python", "handler.py"]
