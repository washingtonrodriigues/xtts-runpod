import os
import sys
import base64
import tempfile
import subprocess
import traceback
import time
import runpod
from TTS.api import TTS

# ===== ENV =====
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["COQUI_COMMERCIAL_LICENSE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

def log(msg: str):
    print(f"[XTTS-HANDLER] {msg}")
    sys.stdout.flush()

# ===== Load model once (global) =====
log("Carregando modelo XTTS v2...")
tts = TTS(model_name=MODEL_NAME, gpu=True)
log("Modelo XTTS v2 carregado!")

# ===== Helper: WAV -> OGG (opus) =====
def wav_to_ogg(wav_path: str) -> str:
    ogg_path = wav_path.replace(".wav", ".ogg")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libopus",
            "-b:a", "32k",
            ogg_path
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return ogg_path

# ===== Handler =====
def handler(job):
    try:
        data = job.get("input", {})

        text = data.get("text")
        language = data.get("language", "pt")
        speaker_b64 = data.get("speaker_wav_base64")

        if not text:
            return {"error": "Campo 'text' é obrigatório"}

        if not speaker_b64:
            return {"error": "Campo 'speaker_wav_base64' é obrigatório"}

        # ---- Speaker WAV ----
        speaker_bytes = base64.b64decode(speaker_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as speaker_file:
            speaker_file.write(speaker_bytes)
            speaker_path = speaker_file.name

        # ---- Output WAV ----
        out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        start = time.time()

        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=out_wav
        )

        generation_time = time.time() - start
        log(f"TTS gerado em {generation_time:.2f}s")

        # ---- WAV -> OGG ----
        out_ogg = wav_to_ogg(out_wav)

        # ---- Base64 OGG ----
        with open(out_ogg, "rb") as f:
            audio_ogg_base64 = base64.b64encode(f.read()).decode("utf-8")

        # ---- Cleanup ----
        os.remove(speaker_path)
        os.remove(out_wav)
        os.remove(out_ogg)

        return {
            "audio_base64": audio_ogg_base64,
            "mimetype": "audio/ogg; codecs=opus",
            "filename": "voice-message.ogg",
            "format": "ogg",
            "generation_time": generation_time,
            "status": "success"
        }

    except Exception as e:
        log("Erro no handler:")
        log(traceback.format_exc())
        return {
            "error": str(e),
            "status": "failed"
        }

# ===== RunPod =====
runpod.serverless.start({"handler": handler})
