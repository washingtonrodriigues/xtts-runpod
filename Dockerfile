FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ===============================
# Variáveis de ambiente (críticas)
# ===============================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Limita threads (RAM + estabilidade)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Evita fragmentação de VRAM
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
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# ===============================
# Python / Pip
# ===============================
RUN pip install --upgrade pip

# PyTorch com CUDA 11.8 (compatível com XTTS v2)
RUN pip install \
    torch==2.2.2+cu118 \
    torchaudio==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# RunPod + Coqui TTS
RUN pip install \
    runpod \
    TTS

# ===============================
# Download do modelo no build
# (reduz cold start)
# ===============================
RUN python - <<EOF
from TTS.api import TTS
TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)
EOF

# ===============================
# Código
# ===============================
COPY handler.py .

# ===============================
# Start
# ===============================
CMD ["python", "handler.py"]
