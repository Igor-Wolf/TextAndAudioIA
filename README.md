# TextToAudioTTS
Transformar texto em audio através de uma api em python utilizando o TTS


Crie a venv pelo cmd dentro do diretório:

py -3.10 -m venv venv310


Ative a venv pelo Shell do vscode:


.\venv310\Scripts\activate


intalar dependencias:


problema com o torch:

python -m pip install --upgrade pip setuptools wheel

pip install --upgrade transformers datasets[audio] accelerate

pip install torch TTS

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.cuda.is_available())"

pip install -r requirements.txt


pip install fastapi[standard]

fazer rodar a fastapi:

fastapi dev main.py --port 8085

$env:Path += ";C:\ffmpeg\bin"
