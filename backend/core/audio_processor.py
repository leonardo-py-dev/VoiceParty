import os
import sys
from typing import Optional
from collections import OrderedDict
import torch
import numpy as np
import librosa
import librosa.sequence
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter1d

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from backend.ml.model import SiameseAudioCRNN
from backend.ml.dataset import load_and_preprocess_audio, audio_to_log_mel, TARGET_SR, TARGET_SAMPLES

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "checkpoints", "siamese_crnn.pt")

class AudioProcessor:
    def __init__(self, model_path: str = CHECKPOINT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseAudioCRNN(n_mels=64, embedding_dim=128).to(self.device)
        self.is_model_loaded = False
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_cache_size: int = 64
        
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"[AVISO] Checkpoint não encontrado em {model_path}. O modelo rodará com pesos iniciais até ser treinado.")

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.is_model_loaded = True
            print(f"[OK] Modelo da Rede Siamesa carregado com sucesso de {model_path}")
        except Exception as e:
            print(f"[ERRO] Erro ao carregar checkpoint do modelo: {e}")

    def clear_cache(self):
        """Limpa o cache em memória RAM."""
        self._cache.clear()

    def preload_catalog(self, catalog_dir: str = None):
        """Pré-carrega todos os arquivos WAV do catálogo em memória RAM para eliminar I/O de disco."""
        if catalog_dir is None:
            catalog_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "clips"))
        if not os.path.exists(catalog_dir):
            return
        for fname in os.listdir(catalog_dir):
            if fname.endswith(".wav") and not fname.startswith("test_attempt"):
                fpath = os.path.join(catalog_dir, fname)
                try:
                    self.get_features(fpath)
                except Exception:
                    pass

    def get_features(self, audio_input) -> dict:
        """
        Retorna dicionário com todos os atributos acústicos e neurais pré-calculados.
        Utiliza cache em RAM quando audio_input é o caminho de um arquivo.
        """
        if isinstance(audio_input, dict) and "embedding" in audio_input and "f0" in audio_input:
            return audio_input

        mtime = None
        if isinstance(audio_input, str):
            if os.path.exists(audio_input):
                mtime = os.path.getmtime(audio_input)
                cached = self._cache.get(audio_input)
                if cached is not None and cached.get("mtime") == mtime:
                    self._cache.move_to_end(audio_input)
                    return cached
                elif cached is not None:
                    self._cache.pop(audio_input, None)
            try:
                y, sr = librosa.load(audio_input, sr=TARGET_SR, mono=True)
            except Exception:
                y = load_and_preprocess_audio(audio_input, target_sr=TARGET_SR, target_samples=TARGET_SAMPLES)
        else:
            y = np.asarray(audio_input, dtype=np.float32)

        # Remove silêncio inicial e final para isolar o trecho falado ativo
        if len(y) > 0:
            y_trim, _ = librosa.effects.trim(y, top_db=25)
            if len(y_trim) == 0:
                y_trim = y
        else:
            y_trim = np.zeros(TARGET_SR, dtype=np.float32)

        max_v = float(np.max(np.abs(y_trim)))
        if max_v > 1e-6:
            y_trim = y_trim / max_v

        duration = float(len(y_trim) / TARGET_SR)

        # 1. Curva de Pitch e máscara de voz ativa
        f0_smooth, voiced = self.extract_pitch_curve(y_trim, sr=TARGET_SR)

        # 2. Envelope de intensidade RMS na fala ativa
        rms = librosa.feature.rms(y=y_trim, hop_length=256)[0]
        if np.max(rms) > 0:
            rms = rms / np.max(rms)
        rms_env = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, max(1, len(rms))), rms)

        # 3. Embedding Neural CRNN com eliminação de viés de silêncio (tiling do áudio ativo)
        if len(y_trim) < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / max(1, len(y_trim))))
            y_neural = np.tile(y_trim, repeats)[:TARGET_SAMPLES]
        else:
            y_neural = y_trim[:TARGET_SAMPLES]

        mel_tensor = audio_to_log_mel(y_neural).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.forward_one(mel_tensor).squeeze(0).cpu().numpy()

        feat = {
            "y": y_trim,
            "duration": duration,
            "f0": f0_smooth,
            "voiced": voiced,
            "rms_env": rms_env,
            "embedding": embedding,
            "mtime": mtime
        }

        if isinstance(audio_input, str):
            self._cache[audio_input] = feat
            self._cache.move_to_end(audio_input)
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        return feat

    def extract_embedding(self, audio_path_or_waveform) -> np.ndarray:
        """
        Extrai o vetor de 128 dimensões da Rede Neural Siamesa.
        """
        feat = self.get_features(audio_path_or_waveform)
        return feat["embedding"]

    def compute_neural_similarity(self, ref_audio, user_audio) -> float:
        """
        Calcula a similaridade da Rede Siamesa (0.0 a 100.0) calibrada para a margem da Triplet Loss.
        Calibração não-linear contínua:
        - dot_product <= 0.15 -> 0.0 (áudio ortogonal / ruído / negativo distante)
        - 0.15 < dot_product < 0.85 -> até 20.0 (áudios díspares com baixa correlação espectral)
        - 0.85 <= dot_product < 0.89 -> transição suave para vozes aproximadas
        - 0.89 <= dot_product <= 1.0 -> 85.0 a 100.0 (imitações fiéis e pequenas variações tonais)
        """
        f_ref = self.get_features(ref_audio)
        f_user = self.get_features(user_audio)

        # Produto escalar (ambos são normalizados L2 na saída do modelo)
        dot_product = float(np.dot(f_ref["embedding"], f_user["embedding"]))
        
        if dot_product <= 0.15:
            score = 0.0
        elif dot_product < 0.85:
            t = (dot_product - 0.15) / (0.85 - 0.15)
            score = 20.0 * (t ** 1.5)
        elif dot_product < 0.89:
            t = (dot_product - 0.85) / (0.89 - 0.85)
            s = 3 * (t ** 2) - 2 * (t ** 3)
            score = 20.0 + 65.0 * s
        else:
            t = (dot_product - 0.89) / (1.0 - 0.89)
            score = np.clip(85.0 + 15.0 * t, 85.0, 100.0)

        return round(float(score), 2)

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

    def compute_pitch_contour_similarity(self, ref_audio, user_audio) -> float:
        """
        Compara o contorno melódico/entonação da fala usando Dynamic Time Warping (DTW) real
        exclusivamente nas regiões sonoras ativas (voiced), eliminando viés de silêncio.
        Normalizado por z-score para avaliar a subida e descida da entonação independente do gênero vocal.
        """
        f_ref = self.get_features(ref_audio)
        f_user = self.get_features(user_audio)

        p_ref, v_ref = f_ref["f0"], f_ref["voiced"]
        p_user, v_user = f_user["f0"], f_user["voiced"]

        # Se nenhum som voiced foi detectado em um dos áudios
        if not np.any(v_ref) or not np.any(v_user):
            return 20.0

        ref_vals = p_ref[v_ref]
        user_vals = p_user[v_user]

        if len(ref_vals) < 3 or len(user_vals) < 3:
            return 20.0

        # Normalização relativa Z-score exclusivamente dos trechos com voz ativa
        ref_norm = (ref_vals - np.mean(ref_vals)) / (np.std(ref_vals) + 1e-6)
        user_norm = (user_vals - np.mean(user_vals)) / (np.std(user_vals) + 1e-6)

        # Dynamic Time Warping real com librosa.sequence.dtw
        D, wp = librosa.sequence.dtw(
            X=ref_norm.reshape(1, -1),
            Y=user_norm.reshape(1, -1),
            metric='euclidean'
        )
        norm_cost = float(D[-1, -1] / max(1, len(wp)))

        # Penalidade proporcional se a duração dos segmentos sonoros for muito díspar
        len_ratio = float(min(len(ref_vals), len(user_vals)) / max(len(ref_vals), len(user_vals)))
        eff_dist = norm_cost + (1.0 - len_ratio) * 0.8

        # Mapeamento exponencial para pontuação 0 a 100
        score = np.exp(-eff_dist * 2.2) * 100.0
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def compute_rhythm_similarity(self, ref_audio, user_audio) -> float:
        """
        Compara a cadência e dinâmica de intensidade da fala usando o envelope RMS
        dos trechos ativos, eliminando diluição por preenchimento de silêncio.
        """
        f_ref = self.get_features(ref_audio)
        f_user = self.get_features(user_audio)

        env_ref = f_ref["rms_env"]
        env_user = f_user["rms_env"]

        # Correlação de Pearson dos envelopes de energia ativos
        correlation = np.corrcoef(env_ref, env_user)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Fator de consistência da duração ativa
        dur_ref = max(0.01, f_ref["duration"])
        dur_user = max(0.01, f_user["duration"])
        dur_ratio = float(min(dur_ref, dur_user) / max(dur_ref, dur_user))

        score = max(0.0, float(correlation)) * 100.0 * (dur_ratio ** 0.5)
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def export_spectrogram(self, y: np.ndarray, n_mels: int = 64, target_frames: int = 60) -> list[list[float]]:
        """
        Gera matriz 2D de Log-Mel Espectrograma (n_mels x target_frames) normalizada (0.0 a 1.0)
        para visualização didática em heatmap no frontend.
        """
        if len(y) == 0:
            return [[0.0] * target_frames for _ in range(n_mels)]

        mel = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_mels=n_mels, n_fft=1024, hop_length=512
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Normalização min-max contínua para escala [0.0, 1.0]
        min_val = float(np.min(log_mel))
        max_val = float(np.max(log_mel))
        range_val = max_val - min_val
        if range_val > 1e-5:
            norm_mel = (log_mel - min_val) / range_val
        else:
            norm_mel = np.zeros_like(log_mel)

        # Downsample/interpolação no eixo do tempo para target_frames
        n_frames = norm_mel.shape[1]
        if n_frames == target_frames:
            resampled = norm_mel
        elif n_frames > 0:
            orig_indices = np.linspace(0, 1, n_frames)
            target_indices = np.linspace(0, 1, target_frames)
            resampled = np.zeros((n_mels, target_frames), dtype=np.float32)
            for i in range(n_mels):
                resampled[i, :] = np.interp(target_indices, orig_indices, norm_mel[i, :])
        else:
            resampled = np.zeros((n_mels, target_frames), dtype=np.float32)

        return [[round(float(val), 3) for val in row] for row in np.clip(resampled, 0.0, 1.0)]

    def export_mfcc(self, y: np.ndarray, n_mfcc: int = 13, target_frames: int = 60) -> list[list[float]]:
        """
        Extrai 13 Coeficientes Cepstrais de Frequência de Mel (MFCCs) usando Librosa,
        normalizados para [0.0, 1.0] para visualização em tempo real no Playground.
        """
        if len(y) == 0:
            return [[0.0] * target_frames for _ in range(n_mfcc)]

        mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
        min_val = float(np.min(mfcc))
        max_val = float(np.max(mfcc))
        range_val = max_val - min_val
        if range_val > 1e-5:
            norm_mfcc = (mfcc - min_val) / range_val
        else:
            norm_mfcc = np.zeros_like(mfcc)

        n_frames = norm_mfcc.shape[1]
        if n_frames == target_frames:
            resampled = norm_mfcc
        elif n_frames > 0:
            orig_indices = np.linspace(0, 1, n_frames)
            target_indices = np.linspace(0, 1, target_frames)
            resampled = np.zeros((n_mfcc, target_frames), dtype=np.float32)
            for i in range(n_mfcc):
                resampled[i, :] = np.interp(target_indices, orig_indices, norm_mfcc[i, :])
        else:
            resampled = np.zeros((n_mfcc, target_frames), dtype=np.float32)

        return [[round(float(val), 3) for val in row] for row in np.clip(resampled, 0.0, 1.0)]

    def export_pitch_visualization(self, ref_feat: dict, user_feat: dict, target_points: int = 60) -> tuple[dict, list[float], list[float]]:
        """
        Gera dados da curva de pitch (F0 em Hz) alinhados para o gráfico de linhas do frontend
        e as listas simplificadas pitch_ref e pitch_usr para compatibilidade legada.
        """
        p_ref = ref_feat.get("f0", np.zeros(1))
        v_ref = ref_feat.get("voiced", np.zeros(1, dtype=bool))
        p_user = user_feat.get("f0", np.zeros(1))
        v_user = user_feat.get("voiced", np.zeros(1, dtype=bool))

        dur_ref = max(0.1, float(ref_feat.get("duration", 1.0)))
        dur_user = max(0.1, float(user_feat.get("duration", 1.0)))
        max_dur = max(dur_ref, dur_user)

        time_axis = np.linspace(0.0, max_dur, target_points)

        # Interpolação para grade temporal uniforme
        def interp_curve(f0_arr, dur):
            if len(f0_arr) == 0:
                return np.zeros(target_points)
            t_orig = np.linspace(0.0, dur, len(f0_arr))
            return np.interp(time_axis, t_orig, f0_arr, left=0.0, right=0.0)

        def interp_voiced(v_arr, dur):
            if len(v_arr) == 0:
                return [False] * target_points
            t_orig = np.linspace(0.0, dur, len(v_arr))
            v_float = np.interp(time_axis, t_orig, v_arr.astype(float), left=0.0, right=0.0)
            return [bool(x > 0.4) for x in v_float]

        pitch_ref_hz = [round(float(v), 1) for v in interp_curve(p_ref, dur_ref)]
        pitch_usr_hz = [round(float(v), 1) for v in interp_curve(p_user, dur_user)]
        v_ref_interp = interp_curve(v_ref.astype(float), dur_ref)
        v_usr_interp = interp_curve(v_user.astype(float), dur_user)
        voiced_mask = [bool(v_ref_interp[i] > 0.4 or v_usr_interp[i] > 0.4) for i in range(target_points)]

        # Pitch simples (downsampled dos arrays originais para compatibilidade com testes legados)
        step_ref = max(1, len(p_ref) // target_points)
        step_usr = max(1, len(p_user) // target_points)
        pitch_ref_simple = [round(float(v), 1) for v in p_ref[::step_ref]]
        pitch_usr_simple = [round(float(v), 1) for v in p_user[::step_usr]]

        pitch_curve_dict = {
            "time_seconds": [round(float(t), 3) for t in time_axis],
            "pitch_ref_hz": pitch_ref_hz,
            "pitch_usr_hz": pitch_usr_hz,
            "voiced_mask": voiced_mask
        }
        return pitch_curve_dict, pitch_ref_simple, pitch_usr_simple

    def export_embeddings_visualization(self, ref_feat: dict, user_feat: dict) -> dict:
        """
        Gera representação dos embeddings neurais de 128 dimensões e a similaridade de cosseno.
        """
        emb_ref = ref_feat.get("embedding", np.zeros(128))
        emb_usr = user_feat.get("embedding", np.zeros(128))

        dot_prod = float(np.dot(emb_ref, emb_usr))
        norm_ref = float(np.linalg.norm(emb_ref))
        norm_usr = float(np.linalg.norm(emb_usr))
        if norm_ref > 1e-6 and norm_usr > 1e-6:
            cosine_sim = dot_prod / (norm_ref * norm_usr)
        else:
            cosine_sim = 0.0

        return {
            "ref_vector": [round(float(v), 4) for v in emb_ref],
            "usr_vector": [round(float(v), 4) for v in emb_usr],
            "cosine_similarity": round(float(np.clip(cosine_sim, -1.0, 1.0)), 4)
        }

    def export_radar_metrics(
        self,
        s_neural: float,
        s_pitch: float,
        s_rhythm: float,
        s_words: Optional[float],
        ref_feat: dict,
        user_feat: dict
    ) -> dict:
        """
        Gera as 5 dimensões para o gráfico Radar/Spider do frontend:
        timbre, pitch, rhythm, speech, energy.
        """
        speech_score = s_words if s_words is not None else round(float((s_neural * 0.5) + (s_pitch * 0.5)), 1)

        rms_ref = ref_feat.get("rms_env", np.zeros(1))
        rms_usr = user_feat.get("rms_env", np.zeros(1))
        energy_corr = float(np.corrcoef(rms_ref, rms_usr)[0, 1])
        if np.isnan(energy_corr) or energy_corr < 0.0:
            energy_score = max(10.0, s_rhythm * 0.7)
        else:
            energy_score = round(float(np.clip(energy_corr * 100.0, 0.0, 100.0)), 1)

        return {
            "timbre": round(float(s_neural), 1),
            "pitch": round(float(s_pitch), 1),
            "rhythm": round(float(s_rhythm), 1),
            "speech": round(float(speech_score), 1),
            "energy": round(float(energy_score), 1)
        }

