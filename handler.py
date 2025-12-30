import os
import base64
import tempfile
import subprocess
import torch
import runpod
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["COQUI_COMMERCIAL_LICENSE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("Carregando modelo XTTS v2...")
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=True
)
print("Modelo carregado!")

def handler(event):
    data = event["input"]

    text = data["text"]
    language = data.get("language", "pt")
    speaker_b64 = data["speaker_wav_base64"]

    # ---------- Speaker ----------
    speaker_bytes = base64.b64decode(speaker_b64)
    speaker_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    speaker_wav.write(speaker_bytes)
    speaker_wav.close()

    # ---------- Output WAV ----------
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    out_ogg = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg").name

    # TTS
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav.name,
        language=language,
        file_path=out_wav
    )

    # ---------- WAV → OGG (opus) ----------
    subprocess.run([
        "ffmpeg", "-y",
        "-i", out_wav,
        "-c:a", "libopus",
        "-b:a", "64k",
        out_ogg
    ], check=True)

    # ---------- Base64 ----------
    with open(out_ogg, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # Cleanup
    os.remove(speaker_wav.name)
    os.remove(out_wav)
    os.remove(out_ogg)

    torch.cuda.empty_cache()

    return {
        "audio_base64": audio_b64,
        "mimetype": "audio/ogg; codecs=opus"
    }

runpod.serverless.start({"handler": handler})
