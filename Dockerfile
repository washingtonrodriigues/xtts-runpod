FROM python:3.10-slim

# Configurar variáveis de ambiente para o TTS
ENV COQUI_TOS_AGREED=1
ENV COQUI_COMMERCIAL_LICENSE=0
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    git \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Criar diretório da aplicação
WORKDIR /app

# Criar diretório para cache do modelo
RUN mkdir -p /app/models /tmp/tts_cache

# Atualizar pip e instalar ferramentas básicas
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar PyTorch CPU primeiro (versão mais estável)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps

# Copiar requirements
COPY requirements.txt .

# Instalar dependências em etapas para melhor debug
RUN pip install --no-cache-dir numpy==1.24.3

RUN pip install --no-cache-dir scipy==1.11.4

RUN pip install --no-cache-dir soundfile==0.12.1

RUN pip install --no-cache-dir librosa==0.10.1

RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0

RUN pip install --no-cache-dir python-multipart==0.0.6 pydantic==2.5.0

RUN pip install --no-cache-dir python-dotenv==1.0.0 pydub==0.25.1

RUN pip install --no-cache-dir transformers==4.35.2

RUN pip install --no-cache-dir runpod==1.6.0

# TTS por último (dependência mais complexa)
RUN pip install --no-cache-dir TTS==0.22.0

# Limpar cache
RUN pip cache purge

# Copiar código da aplicação
COPY handler.py .

# Criar diretório temporário para processamento
RUN mkdir -p /tmp/tts_temp

# PRÉ-BAIXAR O MODELO DURANTE O BUILD
RUN echo "Pré-baixando modelo XTTS-v2 durante o build..." && \
    python -c "from TTS.api import TTS; print('Iniciando download do modelo...'); TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2'); print('Download concluído!')" && \
    echo "Modelo XTTS-v2 pré-baixado com sucesso!"

# Expor porta (não estritamente necessário para serverless, mas mantido para compatibilidade)
EXPOSE 8000

# Comando para executar o handler
CMD ["python", "handler.py"]
