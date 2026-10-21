import os
import torch
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter1d

from backend.ml.model import SiameseAudioCRNN
from backend.ml.dataset import load_and_preprocess_audio, audio_to_log_mel, TARGET_SR

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "checkpoints", "siamese_crnn.pt")

class AudioProcessor:
    def __init__(self, model_path: str = CHECKPOINT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseAudioCRNN(n_mels=64, embedding_dim=128).to(self.device)
        self.is_model_loaded = False
        
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"⚠️ Checkpoint não encontrado em {model_path}. O modelo rodará com pesos iniciais até ser treinado.")

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.is_model_loaded = True
            print(f"✅ Modelo da Rede Siamesa carregado com sucesso de {model_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar checkpoint do modelo: {e}")

    def extract_embedding(self, audio_path_or_waveform) -> np.ndarray:
        """
        Extrai o vetor de 128 dimensões da Rede Neural Siamesa.
        """
        if isinstance(audio_path_or_waveform, str):
            y = load_and_preprocess_audio(audio_path_or_waveform)
        else:
            y = audio_path_or_waveform

        mel_tensor = audio_to_log_mel(y).unsqueeze(0).to(self.device)  # [1, 1, 64, T]
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.forward_one(mel_tensor)
        return embedding.squeeze(0).cpu().numpy()

    def compute_neural_similarity(self, ref_audio_path: str, user_audio_path: str) -> float:
        """
        Calcula a similaridade da Rede Siamesa (0.0 a 100.0).
        """
        emb_ref = self.extract_embedding(ref_audio_path)
        emb_user = self.extract_embedding(user_audio_path)

        # Produto escalar (ambos são normalizados L2)
        dot_product = float(np.dot(emb_ref, emb_user))
        # Mapeia [-1, 1] para porcentagem [0, 100]
        score_0_1 = max(0.0, min(1.0, (dot_product + 1.0) / 2.0))
        return round(score_0_1 * 100.0, 2)

    def extract_pitch_curve(self, y: np.ndarray, sr: int = TARGET_SR) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrai a curva de pitch (F0 em Hz) usando librosa.yin com suavização.
        """
        fmin = librosa.note_to_hz('C2')   # ~65 Hz
        fmax = librosa.note_to_hz('C7')   # ~2093 Hz
        
        # Extrai pitch via YIN
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=256)
        
        # Máscara de voz ativa (onde a energia é suficiente)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        voiced = (rms > 0.02) & np.isfinite(f0)
        
        # Suavização para ruído
        f0_clean = np.where(voiced, f0, 0.0)
        f0_smooth = gaussian_filter1d(f0_clean, sigma=1.5)
        return f0_smooth, voiced

    def compute_pitch_contour_similarity(self, ref_audio_path: str, user_audio_path: str) -> float:
        """
        Compara o contorno melódico/entonação da fala usando DTW normalizado por z-score.
        Isso não penaliza a diferença anatômica de homens vs. mulheres, avaliando
        se a subida e descida da entonação foi fiel!
        """
        y_ref = load_and_preprocess_audio(ref_audio_path)
        y_user = load_and_preprocess_audio(user_audio_path)

        p_ref, v_ref = self.extract_pitch_curve(y_ref)
        p_user, v_user = self.extract_pitch_curve(y_user)

        # Se nenhum som voiced foi detectado
        if not np.any(v_ref) or not np.any(v_user):
            return 30.0

        # Normalização relativa (Z-score dos trechos falados)
        ref_vals = p_ref[v_ref]
        user_vals = p_user[v_user]

        ref_norm = (p_ref - np.mean(ref_vals)) / (np.std(ref_vals) + 1e-6)
        user_norm = (p_user - np.mean(user_vals)) / (np.std(user_vals) + 1e-6)
        ref_norm[~v_ref] = 0.0
        user_norm[~v_user] = 0.0

        # Resample para mesmo tamanho de frames para comparação DTW rápida
        target_len = 100
        ref_interp = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(ref_norm)), ref_norm)
        user_interp = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(user_norm)), user_norm)

        # Distância média absoluta normalizada
        mean_diff = np.mean(np.abs(ref_interp - user_interp))
        # Converter distância para pontuação 0 a 100
        score = np.exp(-mean_diff * 0.8) * 100.0
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def compute_rhythm_similarity(self, ref_audio_path: str, user_audio_path: str) -> float:
        """
        Compara a cadência e dinâmica de pausas da fala usando o envelope RMS.
        """
        y_ref = load_and_preprocess_audio(ref_audio_path)
        y_user = load_and_preprocess_audio(user_audio_path)

        rms_ref = librosa.feature.rms(y=y_ref, hop_length=256)[0]
        rms_user = librosa.feature.rms(y=y_user, hop_length=256)[0]

        # Normaliza envelopes
        if np.max(rms_ref) > 0:
            rms_ref = rms_ref / np.max(rms_ref)
        if np.max(rms_user) > 0:
            rms_user = rms_user / np.max(rms_user)

        target_len = 100
        ref_env = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(rms_ref)), rms_ref)
        user_env = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(rms_user)), rms_user)

        # Correlação de Pearson dos envelopes de energia
        correlation = np.corrcoef(ref_env, user_env)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        score = max(0.0, correlation) * 100.0
        return round(float(score), 2)
