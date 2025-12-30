import os
import sys
import json
import base64
import tempfile
import traceback
from typing import Optional, Dict, Any
import runpod
from TTS.api import TTS
import librosa
import soundfile as sf
import uuid
import time
import shutil
from pathlib import Path

# Configurações do modelo
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_PATH = os.path.expanduser("~/.local/share/tts/" + MODEL_NAME)

# Cache global do modelo TTS
tts_model = None
model_loading = False

# Configurar diretórios temporários
TEMP_DIR = "/tmp/tts_temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def log(message: str):
    """Função de logging para debug"""
    print(f"[XTTS-HANDLER] {message}")
    sys.stdout.flush()


def preload_model():
    """Pré-carrega o modelo durante a inicialização"""
    global tts_model, model_loading
    
    if model_loading:
        log("Modelo já está sendo carregado...")
        return False
    
    if tts_model is not None:
        log("Modelo já está em memória.")
        return True
    
    model_loading = True
    try:
        log("Pré-carregando modelo XTTS-v2...")
        tts_model = TTS(model_name=MODEL_NAME).to("cpu")
        log("Modelo pré-carregado com sucesso!")
        return True
    except Exception as e:
        log(f"Erro ao pré-carregar modelo: {e}")
        return False
    finally:
        model_loading = False


def get_tts_model():
    """Obtém o modelo TTS com cache em memória"""
    global tts_model
    
    if tts_model is not None:
        return tts_model
    
    # Esperar se o modelo está sendo carregado
    max_wait = 60  # 1 minuto (reduzido)
    wait_time = 0
    while model_loading and wait_time < max_wait:
        time.sleep(1)
        wait_time += 1
    
    if tts_model is not None:
        return tts_model
    
    try:
        log("Carregando modelo XTTS-v2 na memória...")
        tts_model = TTS(model_name=MODEL_NAME).to("cpu")
        log("Modelo carregado com sucesso!")
        return tts_model
    except Exception as e:
        log(f"Erro ao carregar modelo: {e}")
        raise


def wait_for_file(path: str, timeout: int = 5) -> bool:
    """Aguarda arquivo ser completamente escrito"""
    start = time.time()
    while True:
        if os.path.exists(path) and os.path.getsize(path) > 44:
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(0.05)


def cleanup_temp_files():
    """Limpa arquivos temporários antigos"""
    try:
        temp_path = Path(TEMP_DIR)
        current_time = time.time()
        
        for file_path in temp_path.glob("*"):
            if file_path.is_file():
                # Remover arquivos com mais de 10 minutos (reduzido)
                if current_time - file_path.stat().st_mtime > 600:
                    try:
                        file_path.unlink()
                        log(f"Removido arquivo temporário antigo: {file_path}")
                    except Exception as e:
                        log(f"Erro ao remover {file_path}: {e}")
    except Exception as e:
        log(f"Erro na limpeza de arquivos temporários: {e}")


def process_tts_request(job: Dict[str, Any]) -> Dict[str, Any]:
    """Processa uma requisição TTS do endpoint /tts-v2"""
    try:
        # Extrair parâmetros da requisição
        job_input = job.get("input", {})
        
        text = job_input.get("text")
        language = job_input.get("language", "pt")
        speaker_wav_base64 = job_input.get("speaker_wav_base64")
        speed = float(job_input.get("speed", 1.3))
        output_format = job_input.get("output_format", "opus")  # Novo parâmetro
        
        if not text:
            return {
                "error": "Texto é obrigatório",
                "status_code": 400
            }
        
        # Limpar arquivos temporários antigos
        cleanup_temp_files()
        
        # Obter modelo TTS (já deve estar pré-carregado)
        tts = get_tts_model()
        
        # Processar arquivo de referência de voz
        if speaker_wav_base64:
            try:
                audio_bytes = base64.b64decode(speaker_wav_base64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as ref:
                    ref.write(audio_bytes)
                    speaker_path = ref.name
            except Exception as e:
                return {
                    "error": f"Erro ao decodificar áudio base64: {e}",
                    "status_code": 400
                }
        else:
            # Retornar erro se não tiver voz padrão (evita problemas)
            return {
                "error": "speaker_wav_base64 é obrigatório para esta implementação serverless",
                "status_code": 400,
                "suggestion": "Forneça um arquivo de áudio em base64 como referência de voz"
            }
        
        # Gerar arquivo de saída
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as temp_out:
            output_path = temp_out.name
        
        log(f"Gerando TTS em {output_path}")
        
        # Gerar TTS com timeout
        start_time = time.time()
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=output_path
        )
        generation_time = time.time() - start_time
        log(f"TTS gerado em {generation_time:.2f}s")
        
        # Validar arquivo gerado
        if not os.path.exists(output_path):
            return {
                "error": "TTS falhou: arquivo não foi criado",
                "status_code": 500
            }
        
        size = os.path.getsize(output_path)
        log(f"WAV gerado: {size} bytes")
        
        if size < 200:  # menor que o header WAV → arquivo vazio
            os.unlink(output_path)
            if speaker_wav_base64:
                os.unlink(speaker_path)
            return {
                "error": f"TTS gerou arquivo inválido ({size} bytes)",
                "status_code": 500
            }
        
        # Aplicar speed se necessário
        if speed != 1.0:
            try:
                audio, sr = librosa.load(output_path, sr=None)
                audio_fast = librosa.effects.time_stretch(audio, speed)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as final:
                    final_path = final.name
                
                sf.write(final_path, audio_fast, sr)
                
                # Substituir arquivo original
                os.unlink(output_path)
                output_path = final_path
                
            except Exception as e:
                log(f"Erro ao aplicar speed: {e}")
                # Continuar com arquivo original se falhar
        
        # Converter para OPUS se solicitado (melhor para WhatsApp/Telegram)
        if output_format.lower() == "opus":
            try:
                audio, sr = librosa.load(output_path, sr=None)
                
                # Usar bitrate mais baixo para WhatsApp/Telegram
                with tempfile.NamedTemporaryFile(suffix=".opus", delete=False, dir=TEMP_DIR) as opus_file:
                    opus_path = opus_file.name
                
                # Converter para OPUS com bitrate otimizado
                sf.write(opus_path, audio, sr, format='OPUS', bitrate=24)
                
                # Ler arquivo OPUS e converter para base64
                with open(opus_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                # Limpar arquivos temporários
                if speaker_wav_base64 and os.path.exists(speaker_path):
                    os.unlink(speaker_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                if os.path.exists(opus_path):
                    os.unlink(opus_path)
                
                return {
                    "audio_base64": audio_base64,
                    "filename": "tts.opus",
                    "content_type": "audio/opus; codecs=opus",
                    "status": "success",
                    "generation_time": generation_time,
                    "format": "opus",
                    "size_estimate": len(audio_base64)
                }
                
            except Exception as e:
                log(f"Erro ao converter para OPUS: {e}")
                # Fallback para WAV se falhar conversão
        
        # Retornar WAV como fallback ou se solicitado
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Limpar arquivos temporários
        if speaker_wav_base64 and os.path.exists(speaker_path):
            os.unlink(speaker_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        return {
            "audio_base64": audio_base64,
            "filename": "tts.wav",
            "content_type": "audio/wav",
            "status": "success",
            "generation_time": generation_time,
            "format": "wav",
            "size_estimate": len(audio_base64)
        }
        
    except Exception as e:
        error_msg = f"Erro no processamento TTS: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        return {
            "error": error_msg,
            "status_code": 500
        }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Handler principal do RunPod Serverless"""
    try:
        # Verificar se o job tem o campo 'input'
        if "input" not in job:
            return {
                "error": "Campo 'input' é obrigatório. Formato esperado: {'input': {...}}",
                "status_code": 400,
                "example": {
                    "input": {
                        "endpoint": "tts-v2",
                        "text": "Texto para sintetizar",
                        "language": "pt",
                        "speed": 1.3,
                        "speaker_wav_base64": "base64_do_audio",
                        "output_format": "opus"  # Novo parâmetro opcional
                    }
                }
            }
        
        # Determinar o tipo de requisição
        job_input = job.get("input", {})
        endpoint = job_input.get("endpoint", "tts-v2")
        
        if endpoint == "tts-v2":
            return process_tts_request(job)
        else:
            return {
                "error": f"Endpoint '{endpoint}' não suportado. Use 'tts-v2'",
                "status_code": 400
            }
            
    except Exception as e:
        error_msg = f"Erro no handler: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        return {
            "error": error_msg,
            "status_code": 500
        }


# Inicialização do servidor RunPod
if __name__ == "__main__":
    log("Iniciando handler XTTS-v2 para RunPod Serverless...")
    
    # Verificar se modelo já existe no filesystem
    if os.path.exists(MODEL_PATH):
        log("Modelo XTTS-v2 encontrado no filesystem.")
    else:
        log("Modelo XTTS-v2 não encontrado. Será baixado na primeira requisição.")
    
    # Tentar pré-carregar modelo
    preload_model()
    
    # Iniciar servidor RunPod
    runpod.serverless.start({"handler": handler})
