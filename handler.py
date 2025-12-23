import runpod
from TTS.api import TTS
import base64
import tempfile
import os

print("Carregando modelo XTTS v2...")
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=True
)
print("Modelo carregado!")

def handler(event):
    """
    event["input"] esperado:
    {
      "text": "Hello world",
      "language": "en",
      "speaker_wav_base64": "..."
    }
    """

    data = event["input"]

    text = data["text"]
    language = data.get("language", "en")
    speaker_b64 = data["speaker_wav_base64"]

    # salvar speaker temporário
    speaker_bytes = base64.b64decode(speaker_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(speaker_bytes)
        speaker_path = f.name

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_path,
        language=language,
        file_path=out_path
    )

    with open(out_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    os.remove(speaker_path)
    os.remove(out_path)

    return {
        "audio_base64": audio_b64
    }

runpod.serverless.start({"handler": handler})
