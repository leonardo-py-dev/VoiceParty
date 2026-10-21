import os
import random
import tempfile
import subprocess
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

TARGET_SR = 16000
TARGET_DURATION = 3.5  # segundos padronizados
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

def convert_to_wav(input_path: str, output_path: str, target_sr: int = TARGET_SR) -> bool:
    """Converte qualquer formato de áudio (incluindo WebM/Opus do navegador) para WAV 16kHz Mono."""
    cmd = [
        FFMPEG_EXE, "-y", "-i", input_path,
        "-ar", str(target_sr), "-ac", "1",
        output_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0 and os.path.exists(output_path)

def load_and_preprocess_audio(file_path: str, target_sr: int = TARGET_SR, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    """
    Carrega o áudio, converte para mono, faz resample para 16kHz, remove silêncio e padroniza a duração.
    Suporta automaticamente áudios do navegador gravados em WebM/Opus.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception:
        # Se o libsoundfile falhar (ex: WebM do MediaRecorder do Chrome), converte via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            if convert_to_wav(file_path, tmp_wav, target_sr):
                y, sr = librosa.load(tmp_wav, sr=target_sr, mono=True)
            else:
                y = np.zeros(target_samples, dtype=np.float32)
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except:
                    pass

    # Remove silêncio no início e fim
    if len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) == 0:
        y = np.zeros(target_samples, dtype=np.float32)

    if len(y) < target_samples:
        pad_width = target_samples - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    else:
        y = y[:target_samples]

    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val
    return y.astype(np.float32)

def audio_to_log_mel(y: np.ndarray, sr: int = TARGET_SR, n_mels: int = N_MELS, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> torch.Tensor:
    """
    Converte um sinal de áudio 1D em tensor Log-Mel Spectrogram 2D [1, n_mels, time_frames].
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = np.clip(log_mel, 0.0, 1.0)
    return torch.from_numpy(log_mel).unsqueeze(0).float()

def fast_speed_shift(y: np.ndarray, rate: float) -> np.ndarray:
    """Modifica velocidade/pitch de forma ultrarrápida usando interpolação linear vetorial."""
    n_samples = len(y)
    new_len = int(n_samples / rate)
    x_old = np.linspace(0, 1, n_samples)
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, y)

    if len(stretched) < n_samples:
        return np.pad(stretched, (0, n_samples - len(stretched)), mode='constant')
    return stretched[:n_samples]

def augment_positive(y: np.ndarray) -> np.ndarray:
    """
    Data Augmentation ultrarrápido para par Positivo (mesma entonação com pequenas variações acústicas).
    """
    y_aug = y.copy()
    
    # 1. Variação leve de velocidade/pitch (0.96x a 1.04x)
    rate = random.uniform(0.96, 1.04)
    y_aug = fast_speed_shift(y_aug, rate)

    # 2. Deslocamento temporal sutil (roll de até 50ms)
    shift = random.randint(-400, 400)
    y_aug = np.roll(y_aug, shift)

    # 3. Ruído de microfone ambiente
    noise = np.random.randn(len(y_aug)) * random.uniform(0.001, 0.006)
    y_aug = y_aug + noise

    # 4. Variação de ganho
    y_aug = y_aug * random.uniform(0.85, 1.15)
    return y_aug.astype(np.float32)

def augment_negative(y: np.ndarray) -> np.ndarray:
    """
    Data Augmentation ultrarrápido para par Negativo (deforma severamente o ritmo ou inverte a fala).
    """
    y_neg = y.copy()
    choice = random.choice(["reverse", "extreme_fast", "extreme_slow", "choppy"])
    
    if choice == "reverse":
        # Fala invertida no tempo
        y_neg = np.flip(y_neg)
    elif choice == "extreme_fast":
        # Ritmo 1.7x acelerado
        y_neg = fast_speed_shift(y_neg, 1.7)
    elif choice == "extreme_slow":
        # Ritmo 0.55x arrastado
        y_neg = fast_speed_shift(y_neg, 0.55)
    else:
        # Pica o áudio em blocos aleatórios
        block_size = int(TARGET_SR * 0.3)
        for i in range(0, len(y_neg) - block_size, block_size * 2):
            y_neg[i:i + block_size] = 0.0

    return y_neg.astype(np.float32)

class AudioTripletDataset(Dataset):
    """
    Dataset para treinamento Triplet (Anchor, Positive, Negative).
    """
    def __init__(self, audio_paths: list[str], samples_per_epoch: int = 160):
        self.audio_paths = audio_paths
        self.samples_per_epoch = samples_per_epoch
        # Carrega e pré-processa uma única vez na inicialização
        self.cached_waveforms = [load_and_preprocess_audio(p) for p in audio_paths]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        num_audios = len(self.cached_waveforms)
        anchor_idx = random.randint(0, num_audios - 1)
        anchor_audio = self.cached_waveforms[anchor_idx]

        # Positivo: mesma fala com variação sutil
        pos_audio = augment_positive(anchor_audio)

        # Negativo: outro áudio diferente (60%) ou áudio deformado (40%)
        if num_audios > 1 and random.random() < 0.6:
            neg_idx = random.choice([i for i in range(num_audios) if i != anchor_idx])
            neg_audio = self.cached_waveforms[neg_idx]
            if random.random() < 0.5:
                neg_audio = augment_positive(neg_audio)
        else:
            neg_audio = augment_negative(anchor_audio)

        # Converte para Espectrogramas Log-Mel
        anchor_mel = audio_to_log_mel(anchor_audio)
        pos_mel = audio_to_log_mel(pos_audio)
        neg_mel = audio_to_log_mel(neg_audio)

        return anchor_mel, pos_mel, neg_mel
