FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ===============================
# Variáveis de ambiente
# ===============================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ===============================
# Sistema
# ===============================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    sox \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# ===============================
# Python
# ===============================
RUN pip install --upgrade pip

# PyTorch CUDA 11.8
RUN pip install \
    torch==2.2.2+cu118 \
    torchaudio==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# RunPod + TTS
RUN pip install \
    runpod \
    TTS \
    soundfile

# ===============================
# Preload do modelo (SEM GPU)
# ===============================
RUN python - <<EOF
from TTS.api import TTS
print("Baixando XTTS v2...")
TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print("Download concluído")
EOF

# ===============================
# Código
# ===============================
COPY handler.py .

CMD ["python", "handler.py"]
