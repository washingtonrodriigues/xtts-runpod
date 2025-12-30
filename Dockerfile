FROM python:3.10-slim

# Configurar variáveis de ambiente para o TTS
ENV COQUI_TOS_AGREED=1
ENV COQUI_COMMERCIAL_LICENSE=0
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Criar diretório da aplicação
WORKDIR /app

# Criar diretório para cache do modelo
RUN mkdir -p /app/models /tmp/tts_cache

# Copiar requirements primeiro para cache do Docker
COPY requirements-serverless.txt .

# Instalar dependências Python com otimizações para serverless
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serverless.txt \
    && pip cache purge

# Copiar código da aplicação
COPY handler.py .

# Copiar arquivo de voz de referência padrão se existir
COPY female_voice.opus* ./ 2>/dev/null || true

# Criar diretório temporário para processamento
RUN mkdir -p /tmp/tts_temp

# Expor porta (não estritamente necessário para serverless, mas mantido para compatibilidade)
EXPOSE 8000

# Comando para executar o handler
CMD ["python", "handler.py"]
