import librosa
import soundfile as sf

def convert_mp3_to_wav(mp3_path, wav_path, target_sr=22050):
    audio, sr = librosa.load(mp3_path, sr=target_sr, mono=True)  # carrega como mono
    sf.write(wav_path, audio, samplerate=target_sr)


convert_mp3_to_wav("assets/demi.mp3", "assets/demi.wav")
