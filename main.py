from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from typing import Literal
from TTS.api import TTS
import numpy as np
import soundfile as sf
import torch
import io
import tempfile
import os
from pathlib import Path

#--------------------------

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import os
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"



#--------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pode colocar o domínio exato aqui, como "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],  # Ou pode restringir métodos, ex: ["GET", "POST"]
    allow_headers=["*"],
)

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

@app.post("/create-audio")
# def create_audio(text: str, language: Literal["en", "pt", "es"] = "pt"):

# def create_audio(text: str, language: Literal["en", "pt", "es"] = "pt", voice_to_be_cloned: UploadFile = File()):

# para requisições do postman ou outros sites, para passar pelo body os valores
def create_audio(
     text: str = Form(...),
     language: Literal["en", "pt", "es"] = Form("pt"),
    #  voice_to_be_cloned: UploadFile = File(...)
):

    voice_path = Path("./assets/freya.wav")
    with open(voice_path, "rb") as f:
        voice_to_be_cloned = f.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_clone_voice:
        # temp_clone_voice.write(voice_to_be_cloned.file.read())
        temp_clone_voice.write(voice_to_be_cloned)
        temp_clone_voice_path = temp_clone_voice.name

    try:
        text = text.replace("*", "").replace("\n", "").replace('"', "").replace("...",".")
        text = text.replace("\\", "").replace("nn", "")
        text = text.replace('.', ';\n')
        audio = tts.tts(
            text=text,
            speaker_wav=temp_clone_voice_path,
            language=language
        )
    finally:
        os.remove(temp_clone_voice_path)

    audio_array = np.array(audio, dtype=np.float32)
    audio_stream = io.BytesIO()
    sf.write(audio_stream, audio_array, samplerate=22050, format="WAV")
    audio_stream.seek(0)

    return StreamingResponse(
        audio_stream,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'}
    )


#--------------------------------------------------------------------------------------------------------  Audio to text



# Carrega o modelo e pipeline uma vez só
device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device1)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device1,
)

@app.post("/create-text")
# def create_audio(audio_to_convert: UploadFile = File(...)):
def create_audio(
      audio_to_convert: UploadFile = File(...)
):
    audio_bytes = audio_to_convert.file.read()
    # result = pipe(audio_bytes)
    result = pipe(audio_bytes, return_timestamps=True)

    return JSONResponse(content={"text": result["text"]})
