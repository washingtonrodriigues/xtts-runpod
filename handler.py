import os
import sys
import json
import base64
import tempfile
import traceback
import subprocess
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import runpod
from TTS.api import TTS
import librosa
import soundfile as sf
import numpy as np
import uuid
import time
import shutil
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

# Configurações
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_PATH = os.path.expanduser("~/.local/share/tts/" + MODEL_NAME)
TEMP_DIR = "/tmp/tts_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Cache do modelo
tts_model = None
model_loading = False

# Configuração S3/MinIO
S3_ENABLED = all([
    os.getenv('S3_ENDPOINT'),
    os.getenv('S3_ACCESS_KEY'),
    os.getenv('S3_SECRET_KEY'),
    os.getenv('S3_BUCKET')
])

# MinIO usa URLs públicas customizadas
MINIO_PUBLIC_URL = os.getenv('MINIO_PUBLIC_URL')  # Ex: https://minio.yourdomain.com

s3_client = None


def log(message: str):
    """Função de logging"""
    print(f"[XTTS-HANDLER] {message}")
    sys.stdout.flush()


def get_s3_client():
    """Obtém cliente S3/MinIO (lazy loading)"""
    global s3_client
    
    if not S3_ENABLED:
        return None
    
    if s3_client is None:
        try:
            # MinIO requer estas configs específicas
            s3_client = boto3.client('s3',
                endpoint_url=os.getenv('S3_ENDPOINT'),
                aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
                aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
                region_name=os.getenv('S3_REGION', 'us-east-1'),  # MinIO aceita qualquer region
                config=boto3.session.Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}  # MinIO usa path-style
                )
            )
            
            # Testar conexão
            s3_client.head_bucket(Bucket=os.getenv('S3_BUCKET'))
            log("Cliente MinIO inicializado com sucesso")
            
        except Exception as e:
            log(f"Erro ao inicializar MinIO: {e}")
            log(f"Endpoint: {os.getenv('S3_ENDPOINT')}")
            log(f"Bucket: {os.getenv('S3_BUCKET')}")
            return None
    
    return s3_client


def upload_to_s3(file_path: str, content_type: str = "audio/ogg") -> Optional[Dict[str, Any]]:
    """
    Faz upload do arquivo para MinIO e retorna URL
    
    MinIO pode retornar URL pública direta OU URL pré-assinada
    dependendo da configuração do bucket (público ou privado)
    """
    try:
        client = get_s3_client()
        if not client:
            log("MinIO não configurado, usando fallback base64")
            return None
        
        bucket = os.getenv('S3_BUCKET')
        use_public_url = os.getenv('MINIO_USE_PUBLIC_URL', 'false').lower() == 'true'
        
        # Gerar nome único com estrutura de pastas por data
        ext = os.path.splitext(file_path)[1]
        date_prefix = datetime.now().strftime('%Y%m%d')
        filename = f"tts/{date_prefix}/{uuid.uuid4()}{ext}"
        
        # Upload
        log(f"Fazendo upload para MinIO: {filename}")
        
        extra_args = {
            'ContentType': content_type,
            'CacheControl': 'max-age=3600'
        }
        
        # Se bucket for público, adicionar ACL
        if use_public_url:
            extra_args['ACL'] = 'public-read'
        
        client.upload_file(
            file_path,
            bucket,
            filename,
            ExtraArgs=extra_args
        )
        
        file_size = os.path.getsize(file_path)
        
        # Escolher tipo de URL baseado na configuração
        if use_public_url and MINIO_PUBLIC_URL:
            # URL pública direta (não expira)
            url = f"{MINIO_PUBLIC_URL}/{bucket}/{filename}"
            expires_in = None
            log(f"Upload concluído (URL pública): {filename} ({file_size} bytes)")
        else:
            # URL pré-assinada (expira em X segundos)
            expires_in = int(os.getenv('MINIO_URL_EXPIRY', '3600'))  # Padrão: 1 hora
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': filename},
                ExpiresIn=expires_in
            )
            log(f"Upload concluído (URL pré-assinada): {filename} ({file_size} bytes)")
        
        return {
            "url": url,
            "key": filename,
            "expires_in": expires_in,
            "size_bytes": file_size,
            "is_public": use_public_url
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        log(f"Erro MinIO (ClientError - {error_code}): {e}")
        
        # Erros comuns do MinIO
        if error_code == 'NoSuchBucket':
            log(f"ERRO: Bucket '{bucket}' não existe no MinIO!")
        elif error_code == 'InvalidAccessKeyId':
            log("ERRO: Access Key inválida!")
        elif error_code == 'SignatureDoesNotMatch':
            log("ERRO: Secret Key incorreta!")
        
        return None
        
    except Exception as e:
        log(f"Erro ao fazer upload para MinIO: {e}")
        log(traceback.format_exc())
        return None


def get_device():
    """Detecta GPU"""
    import torch
    if torch.cuda.is_available():
        log("GPU detectada, usando CUDA")
        return "cuda"
    else:
        log("GPU não disponível, usando CPU")
        return "cpu"


def generate_waveform(audio_path: str, num_samples: int = 64) -> List[int]:
    """Gera waveform simplificado"""
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        chunk_size = max(1, len(audio) // num_samples)
        
        waveform = []
        for i in range(num_samples):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio))
            
            if start_idx >= len(audio):
                waveform.append(0)
                continue
            
            chunk = audio[start_idx:end_idx]
            rms = np.sqrt(np.mean(chunk**2))
            amplitude = int(min(100, rms * 300))
            waveform.append(amplitude)
        
        return smooth_waveform(waveform)
        
    except Exception as e:
        log(f"Erro ao gerar waveform: {e}")
        return [50] * num_samples


def smooth_waveform(waveform: List[int], window_size: int = 3) -> List[int]:
    """Suaviza waveform"""
    if len(waveform) < window_size:
        return waveform
    
    smoothed = []
    for i in range(len(waveform)):
        start = max(0, i - window_size // 2)
        end = min(len(waveform), i + window_size // 2 + 1)
        smoothed.append(int(np.mean(waveform[start:end])))
    
    return smoothed


def get_audio_duration(audio_path: str) -> float:
    """Obtém duração do áudio"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return round(duration, 2)
    except Exception as e:
        log(f"Erro ao obter duração: {e}")
        try:
            info = sf.info(audio_path)
            return round(info.duration, 2)
        except:
            return 0.0


def preload_model():
    """Pré-carrega modelo"""
    global tts_model, model_loading
    
    if model_loading or tts_model is not None:
        return True
    
    model_loading = True
    try:
        device = get_device()
        log(f"Pré-carregando modelo XTTS-v2 em {device}...")
        tts_model = TTS(model_name=MODEL_NAME, gpu=False).to(device)
        log("Modelo pré-carregado!")
        return True
    except Exception as e:
        log(f"Erro ao pré-carregar: {e}")
        return False
    finally:
        model_loading = False


def get_tts_model():
    """Obtém modelo TTS com cache"""
    global tts_model
    
    if tts_model is not None:
        return tts_model
    
    # Esperar se carregando
    wait_time = 0
    while model_loading and wait_time < 60:
        time.sleep(1)
        wait_time += 1
    
    if tts_model is not None:
        return tts_model
    
    device = get_device()
    log(f"Carregando modelo em {device}...")
    tts_model = TTS(model_name=MODEL_NAME, gpu=False).to(device)
    return tts_model


def convert_to_ogg_opus(input_path: str, output_path: str) -> bool:
    """Converte para OGG Opus"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:a', 'libopus',
            '-ac', '1',
            '-ar', '16000',
            '-b:a', '24k',
            '-vbr', 'on',
            '-compression_level', '10',
            '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            log(f"Erro ffmpeg: {result.stderr.decode()}")
            return False
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            return False
        
        log(f"Conversão OGG OK: {os.path.getsize(output_path)} bytes")
        return True
        
    except Exception as e:
        log(f"Erro ao converter: {e}")
        return False


def cleanup_temp_files():
    """Limpa arquivos temporários antigos"""
    try:
        for file_path in Path(TEMP_DIR).glob("*"):
            if file_path.is_file() and time.time() - file_path.stat().st_mtime > 600:
                file_path.unlink()
    except Exception as e:
        log(f"Erro na limpeza: {e}")


def process_tts_request(job: Dict[str, Any]) -> Dict[str, Any]:
    """Processa requisição TTS com upload S3"""
    try:
        job_input = job.get("input", {})
        
        text = job_input.get("text")
        language = job_input.get("language", "pt")
        speaker_wav_base64 = job_input.get("speaker_wav_base64")
        speed = float(job_input.get("speed", 1.3))
        output_format = job_input.get("output_format", "ogg")
        waveform_samples = int(job_input.get("waveform_samples", 64))
        use_base64 = job_input.get("use_base64", False)  # Forçar base64 se necessário
        
        if not text:
            return {"error": "Texto é obrigatório", "status_code": 400}
        
        cleanup_temp_files()
        tts = get_tts_model()
        
        # Processar referência de voz
        if not speaker_wav_base64:
            return {
                "error": "speaker_wav_base64 é obrigatório",
                "status_code": 400
            }
        
        audio_bytes = base64.b64decode(speaker_wav_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as ref:
            ref.write(audio_bytes)
            speaker_path = ref.name
        
        # Gerar TTS
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as temp_out:
            output_path = temp_out.name
        
        start_time = time.time()
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=output_path
        )
        generation_time = time.time() - start_time
        log(f"TTS gerado em {generation_time:.2f}s")
        
        # Validar
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 200:
            return {"error": "TTS falhou", "status_code": 500}
        
        # Aplicar speed
        processed_path = output_path
        if speed != 1.0 and abs(speed - 1.0) > 0.01:
            try:
                log(f"Aplicando speed: {speed}x")
                audio, sr = librosa.load(output_path, sr=None, dtype=np.float32)
                audio_fast = librosa.effects.time_stretch(y=audio, rate=speed)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR) as speed_file:
                    speed_path = speed_file.name
                
                sf.write(speed_path, audio_fast, sr, subtype='PCM_16')
                processed_path = speed_path
                log(f"Speed aplicado: {len(audio)/sr:.2f}s → {len(audio_fast)/sr:.2f}s")
            except Exception as e:
                log(f"Erro no speed: {e}, usando original")
        
        # Gerar metadados
        waveform = generate_waveform(processed_path, num_samples=waveform_samples)
        audio_duration = get_audio_duration(processed_path)
        
        # Converter para OGG
        final_path = processed_path
        if output_format.lower() == "ogg":
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False, dir=TEMP_DIR) as ogg_file:
                ogg_path = ogg_file.name
            
            if convert_to_ogg_opus(processed_path, ogg_path):
                final_path = ogg_path
                content_type = "audio/ogg; codecs=opus"
                filename = "voice-message.ogg"
            else:
                content_type = "audio/wav"
                filename = "tts.wav"
        else:
            content_type = "audio/wav"
            filename = "tts.wav"
        
        # Decisão: S3 ou Base64
        response = {
            "status": "success",
            "generation_time": generation_time,
            "format": output_format,
            "waveform": waveform,
            "audio_duration": audio_duration,
            "speed_applied": speed
        }
        
        # Tentar upload S3 primeiro (se não forçar base64)
        if not use_base64 and S3_ENABLED:
            s3_result = upload_to_s3(final_path, content_type)
            
            if s3_result:
                response.update({
                    "delivery_method": "s3_url",
                    "audio_url": s3_result["url"],
                    "url_expires_in": s3_result["expires_in"],
                    "size_bytes": s3_result["size_bytes"]
                })
            else:
                # Fallback para base64
                log("S3 falhou, usando base64")
                use_base64 = True
        
        # Base64 (fallback ou forçado)
        if use_base64 or not S3_ENABLED:
            with open(final_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response.update({
                "delivery_method": "base64",
                "file": {
                    "mimetype": content_type,
                    "filename": filename,
                    "data": audio_base64
                },
                "size_bytes": len(audio_base64) * 3 // 4
            })
        
        # Limpar arquivos temporários
        for temp_file in [speaker_path, output_path, processed_path, final_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        return response
        
    except Exception as e:
        return {
            "error": f"Erro: {str(e)}\n{traceback.format_exc()}",
            "status_code": 500
        }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Handler principal"""
    try:
        if "input" not in job:
            return {"error": "Campo 'input' obrigatório", "status_code": 400}
        
        endpoint = job.get("input", {}).get("endpoint", "tts-v2")
        
        if endpoint == "tts-v2":
            return process_tts_request(job)
        else:
            return {"error": f"Endpoint '{endpoint}' não suportado", "status_code": 400}
            
    except Exception as e:
        return {"error": str(e), "status_code": 500}


if __name__ == "__main__":
    log("Iniciando handler XTTS-v2 (MinIO optimized)...")
    log(f"MinIO habilitado: {S3_ENABLED}")
    
    if S3_ENABLED:
        log(f"Endpoint: {os.getenv('S3_ENDPOINT')}")
        log(f"Bucket: {os.getenv('S3_BUCKET')}")
        log(f"URL Pública: {MINIO_PUBLIC_URL or 'Não configurada (usando URLs pré-assinadas)'}")
        log(f"Modo público: {os.getenv('MINIO_USE_PUBLIC_URL', 'false')}")
    else:
        log("MinIO não configurado - usando Base64")
    
    # Verificar ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        log("✓ ffmpeg encontrado")
    except Exception as e:
        log(f"✗ ffmpeg não encontrado: {e}")
    
    preload_model()
    runpod.serverless.start({"handler": handler})
