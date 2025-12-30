import os
import sys
import json
import base64
import tempfile
import traceback
import subprocess
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


def get_device():
    """Detecta se GPU está disponível"""
    import torch
    if torch.cuda.is_available():
        log("GPU detectada, usando CUDA")
        return "cuda"
    else:
        log("GPU não disponível, usando CPU")
        return "cpu"


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
        device = get_device()
        log(f"Pré-carregando modelo XTTS-v2 em {device}...")
        tts_model = TTS(model_name=MODEL_NAME, gpu=False).to(device)
        log(f"Modelo pré-carregado com sucesso em {device}!")
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
    max_wait = 60  # 1 minuto
    wait_time = 0
    while model_loading and wait_time < max_wait:
        time.sleep(1)
        wait_time += 1
    
    if tts_model is not None:
        return tts_model
    
    try:
        device = get_device()
        log(f"Carregando modelo XTTS-v2 em {device}...")
        tts_model = TTS(model_name=MODEL_NAME, gpu=False).to(device)
        log(f"Modelo carregado com sucesso em {device}!")
        return tts_model
    except Exception as e:
        log(f"Erro ao carregar modelo: {e}")
        raise


def convert_to_ogg_opus(input_path: str, output_path: str) -> bool:
    """
    Converte arquivo de áudio para OGG Opus usando ffmpeg.
    Formato compatível com WhatsApp.
    """
    try:
        # Comando ffmpeg para converter para OGG Opus
        # -ac 1: mono (WhatsApp prefere mono)
        # -ar 16000: sample rate 16kHz (ideal para voz)
        # -b:a 24k: bitrate de 24kbps (bom para voz)
        # -vbr on: variable bitrate
        # -compression_level 10: máxima compressão
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:a', 'libopus',
            '-ac', '1',
            '-ar', '16000',
            '-b:a', '24k',
            '-vbr', 'on',
            '-compression_level', '10',
            '-y',  # sobrescrever arquivo se existir
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        if result.returncode != 0:
            log(f"Erro no ffmpeg: {result.stderr.decode()}")
            return False
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            log("Arquivo OGG não foi criado ou está vazio")
            return False
        
        log(f"Conversão OGG Opus bem-sucedida: {os.path.getsize(output_path)} bytes")
        return True
        
    except subprocess.TimeoutExpired:
        log("Timeout na conversão ffmpeg")
        return False
    except Exception as e:
        log(f"Erro ao converter para OGG Opus: {e}")
        return False


def cleanup_temp_files():
    """Limpa arquivos temporários antigos"""
    try:
        temp_path = Path(TEMP_DIR)
        current_time = time.time()
        
        for file_path in temp_path.glob("*"):
            if file_path.is_file():
                # Remover arquivos com mais de 10 minutos
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
        output_format = job_input.get("output_format", "ogg")
        
        if not text:
            return {
                "error": "Texto é obrigatório",
                "status_code": 400
            }
        
        # Limpar arquivos temporários antigos
        cleanup_temp_files()
        
        # Obter modelo TTS
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
            return {
                "error": "speaker_wav_base64 é obrigatório para esta implementação serverless",
                "status_code": 400,
                "suggestion": "Forneça um arquivo de áudio em base64 como referência de voz"
            }
        
        # Gerar arquivo de saída WAV temporário
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as temp_out:
            output_path = temp_out.name
        
        log(f"Gerando TTS em {output_path}")
        
        # Gerar TTS
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
        
        if size < 200:
            os.unlink(output_path)
            if speaker_wav_base64:
                os.unlink(speaker_path)
            return {
                "error": f"TTS gerou arquivo inválido ({size} bytes)",
                "status_code": 500
            }
        
        # Aplicar speed se necessário
        processed_path = output_path
        if speed != 1.0:
            try:
                audio, sr = librosa.load(output_path, sr=None)
                audio_fast = librosa.effects.time_stretch(audio, rate=speed)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as final:
                    final_path = final.name
                
                sf.write(final_path, audio_fast, sr)
                
                # Usar arquivo processado
                processed_path = final_path
                log(f"Speed aplicado: {speed}x")
                
            except Exception as e:
                log(f"Erro ao aplicar speed: {e}, usando arquivo original")
        
        # Converter para OGG Opus se solicitado
        if output_format.lower() == "ogg":
            try:
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False, dir=TEMP_DIR) as ogg_file:
                    ogg_path = ogg_file.name
                
                # Converter usando ffmpeg
                if not convert_to_ogg_opus(processed_path, ogg_path):
                    raise Exception("Falha na conversão para OGG Opus")
                
                # Ler arquivo OGG e converter para base64
                with open(ogg_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                # Limpar arquivos temporários
                if speaker_wav_base64 and os.path.exists(speaker_path):
                    os.unlink(speaker_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                if processed_path != output_path and os.path.exists(processed_path):
                    os.unlink(processed_path)
                if os.path.exists(ogg_path):
                    os.unlink(ogg_path)
                
                # Retornar no formato esperado pela API do WhatsApp
                return {
                    "file": {
                        "mimetype": "audio/ogg; codecs=opus",
                        "filename": "voice-message.ogg",
                        "data": audio_base64
                    },
                    "session": "Vivo",
                    "convert": False,
                    "status": "success",
                    "generation_time": generation_time,
                    "format": "ogg_opus",
                    "size_bytes": len(audio_base64) * 3 // 4  # tamanho aproximado em bytes
                }
                
            except Exception as e:
                log(f"Erro ao converter para OGG Opus: {e}")
                # Continuar para fallback WAV
        
        # Fallback para WAV
        with open(processed_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Limpar arquivos temporários
        if speaker_wav_base64 and os.path.exists(speaker_path):
            os.unlink(speaker_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        if processed_path != output_path and os.path.exists(processed_path):
            os.unlink(processed_path)
        
        return {
            "audio_base64": audio_base64,
            "filename": "tts.wav",
            "content_type": "audio/wav",
            "status": "success",
            "generation_time": generation_time,
            "format": "wav",
            "size_bytes": len(audio_base64) * 3 // 4
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
                        "output_format": "ogg"
                    }
                }
            }
        
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
    
    if os.path.exists(MODEL_PATH):
        log("Modelo XTTS-v2 encontrado no filesystem.")
    else:
        log("Modelo XTTS-v2 não encontrado. Será baixado na primeira requisição.")
    
    # Verificar se ffmpeg está disponível
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        log("ffmpeg encontrado e funcional")
    except Exception as e:
        log(f"AVISO: ffmpeg não encontrado ou não funcional: {e}")
        log("Conversão para OGG Opus não estará disponível")
    
    preload_model()
    runpod.serverless.start({"handler": handler})
