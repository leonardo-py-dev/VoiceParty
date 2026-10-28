import os
import sys
import pytest
import numpy as np
import librosa

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.core.audio_processor import AudioProcessor
from backend.core.scorer import VoiceScorer

CLIPS_DIR = os.path.join(ROOT_DIR, "backend", "data", "clips")
BIRL_PATH = os.path.join(CLIPS_DIR, "ta_saindo_da_jaula.wav")


@pytest.fixture(scope="module")
def shared_scorer():
    processor = AudioProcessor()
    return VoiceScorer(audio_processor=processor)


@pytest.fixture(scope="module")
def birl_audio():
    assert os.path.exists(BIRL_PATH), f"Arquivo não encontrado: {BIRL_PATH}"
    y, sr = librosa.load(BIRL_PATH, sr=16000, mono=True)
    return y, sr


class TestWhiteNoiseAdversarial:
    """Validação adversarial empírica contra ruído branco de alta intensidade e SNR degradado."""

    @pytest.mark.parametrize("sigma", [0.05, 0.3, 1.0, 5.0])
    def test_white_noise_rejection(self, shared_scorer, birl_audio, sigma):
        """Ruído gaussiano puro deve ser rejeitado com pontuação total baixa (< 30) e neural zerado."""
        y_birl, _ = birl_audio
        np.random.seed(42)
        noise = (np.random.randn(len(y_birl)) * sigma).astype(np.float32)

        res = shared_scorer.evaluate_imitation(BIRL_PATH, noise)

        assert np.isfinite(res.total_score), f"Score total com NaN/Inf para sigma={sigma}"
        assert np.isfinite(res.neural_score), f"Score neural com NaN/Inf para sigma={sigma}"
        assert np.isfinite(res.pitch_score), f"Score pitch com NaN/Inf para sigma={sigma}"
        assert np.isfinite(res.rhythm_score), f"Score ritmo com NaN/Inf para sigma={sigma}"

        assert res.total_score < 30.0, f"Ruído sigma={sigma} deve ter nota < 30, obteve: {res.total_score}"
        assert res.neural_score == 0.0, f"Similaridade neural com ruído puro deve ser 0.0, obteve: {res.neural_score}"

    def test_additive_noise_snr_degradation(self, shared_scorer, birl_audio):
        """Adição progressiva de ruído deve degradar monotonicamente a pontuação sem falhas numéricas."""
        y_birl, _ = birl_audio
        np.random.seed(123)
        noise = np.random.randn(len(y_birl)).astype(np.float32)
        signal_power = np.mean(y_birl ** 2)
        noise_power = np.mean(noise ** 2)

        scores_by_snr = {}
        for snr_db in [20, 10, 0, -5]:
            k = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            y_noisy = (y_birl + noise * k).astype(np.float32)
            res = shared_scorer.evaluate_imitation(BIRL_PATH, y_noisy)
            scores_by_snr[snr_db] = res.total_score
            assert np.isfinite(res.total_score)

        # 20dB (sutil) deve pontuar mais que -5dB (ruído predominante)
        assert scores_by_snr[20] > scores_by_snr[10] >= scores_by_snr[0] >= scores_by_snr[-5]
        assert scores_by_snr[-5] < 35.0, f"SNR negativo deve ter score < 35, obteve {scores_by_snr[-5]}"


class TestTemporalReversalAdversarial:
    """Validação adversarial de inversão temporal (áudio reproduzido de trás para frente)."""

    def test_temporal_reversal_penalty(self, shared_scorer, birl_audio):
        """Áudio invertido no tempo deve sofrer penalização severa no ritmo (< 15) e pitch (< 55)."""
        y_birl, _ = birl_audio
        y_reversed = y_birl[::-1].copy().astype(np.float32)

        res = shared_scorer.evaluate_imitation(BIRL_PATH, y_reversed)

        assert np.isfinite(res.total_score)
        assert np.isfinite(res.rhythm_score)
        assert np.isfinite(res.pitch_score)

        # Ritmo deve cair quase a zero pois o envelope temporal foi espelhado
        assert res.rhythm_score < 15.0, f"Ritmo para áudio invertido deve ser < 15, obteve: {res.rhythm_score}"
        # Pitch DTW deve ser penalizado pois o alinhamento monotônico sofre incompatibilidade
        assert res.pitch_score < 55.0, f"Pitch para áudio invertido deve ser < 55, obteve: {res.pitch_score}"
        # Nota geral deve sofrer queda substantiva frente ao original (100.0)
        assert res.total_score < 75.0, f"Nota geral deve ser severamente reduzida, obteve: {res.total_score}"


class TestPureSineWavesAdversarial:
    """Validação adversarial contra tons senoidais puros e frequências fora de alcance vocal."""

    @pytest.mark.parametrize("freq", [50, 100, 220, 440, 880, 1000, 3000])
    def test_pure_sine_wave_rejection(self, shared_scorer, birl_audio, freq):
        """Ondas senoidais puras devem ter nota neural zerada e nota total < 30."""
        y_birl, sr = birl_audio
        duration = len(y_birl) / sr
        t = np.linspace(0, duration, len(y_birl), endpoint=False)
        sine_wave = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)

        res = shared_scorer.evaluate_imitation(BIRL_PATH, sine_wave)

        assert np.isfinite(res.total_score)
        assert np.isfinite(res.neural_score)
        assert res.total_score < 30.0, f"Tom puro {freq}Hz deve ter total < 30, obteve: {res.total_score}"
        assert res.neural_score == 0.0, f"Tom puro {freq}Hz deve ter neural 0.0, obteve: {res.neural_score}"

    def test_inverted_pitch_contour(self, shared_scorer, birl_audio):
        """Inversão vertical do contorno melódico (melodia espelhada) deve derrubar o pitch DTW."""
        processor = shared_scorer.processor
        feat = processor.get_features(BIRL_PATH)

        feat_inv = dict(feat)
        feat_inv["f0"] = feat["f0"].copy()
        voiced = feat["voiced"]
        mean_voiced = np.mean(feat["f0"][voiced])
        # Espelha o contorno: subidas viram descidas e vice-versa
        feat_inv["f0"][voiced] = 2 * mean_voiced - feat["f0"][voiced]

        pitch_score = processor.compute_pitch_contour_similarity(feat, feat_inv)
        assert pitch_score < 25.0, f"Contorno de pitch espelhado deve ter score < 25, obteve: {pitch_score}"


class TestPitchShiftsAdversarial:
    """Validação adversarial de transposição tonal intencional (+-1, +-5, +-12 semitons)."""

    @pytest.mark.parametrize("semitones", [1, -1, 5, -5, 12, -12])
    def test_pitch_shift_consistency(self, shared_scorer, birl_audio, semitones):
        """Shifts de tom devem manter integridade numérica e decrescer coerentemente."""
        y_birl, sr = birl_audio
        y_shifted = librosa.effects.pitch_shift(y_birl, sr=sr, n_steps=semitones).astype(np.float32)

        res = shared_scorer.evaluate_imitation(BIRL_PATH, y_shifted)

        assert np.isfinite(res.total_score)
        assert np.isfinite(res.neural_score)
        assert np.isfinite(res.pitch_score)
        assert np.isfinite(res.rhythm_score)

        # Transposição sutil (+-1 semitom) deve manter nota aceitável (> 70)
        if abs(semitones) == 1:
            assert res.total_score >= 70.0, f"Shift {semitones:+d} deve ter total >= 70, obteve: {res.total_score}"
            assert res.neural_score >= 85.0
        # Transposição extrema (-12 semitons) deve sofrer forte penalização (< 50)
        elif semitones == -12:
            assert res.total_score < 50.0, f"Shift -12 deve ter total < 50, obteve: {res.total_score}"
            assert res.neural_score < 20.0


class TestNumericalStabilityAndEdgeCases:
    """Validação extrema de estabilidade numérica, silêncio, DC offset e Dirac delta."""

    def test_pure_silence_inputs(self, shared_scorer):
        """Silêncio absoluto em diferentes durações deve retornar pontuação baixa sem crash."""
        for duration_s in [0.0, 0.05, 1.0, 3.5]:
            n_samples = int(duration_s * 16000)
            silence = np.zeros(n_samples, dtype=np.float32)
            res = shared_scorer.evaluate_imitation(BIRL_PATH, silence)

            assert np.isfinite(res.total_score)
            assert res.total_score < 20.0, f"Silêncio ({duration_s}s) deve ter total < 20, obteve: {res.total_score}"
            assert res.neural_score == 0.0
            assert res.rhythm_score == 0.0

    def test_dc_offset_input(self, shared_scorer):
        """Sinal com DC offset constante puro não deve gerar divisão por zero nem travar o DTW."""
        dc = np.ones(16000, dtype=np.float32) * 0.5
        res = shared_scorer.evaluate_imitation(BIRL_PATH, dc)

        assert np.isfinite(res.total_score)
        assert res.total_score < 25.0
        assert res.neural_score == 0.0

    def test_dirac_delta_input(self, shared_scorer):
        """Impulso unitário Dirac delta isolado deve ser tratado sem erro."""
        dirac = np.zeros(16000, dtype=np.float32)
        dirac[500] = 1.0
        res = shared_scorer.evaluate_imitation(BIRL_PATH, dirac)

        assert np.isfinite(res.total_score)
        assert res.total_score < 30.0

    def test_extreme_amplitudes(self, shared_scorer, birl_audio):
        """Sinais com amplitude extrema (10^6 e 10^-10) devem ser normalizados sem estourar float."""
        y_birl, _ = birl_audio

        res_huge = shared_scorer.evaluate_imitation(BIRL_PATH, y_birl * 1e6)
        assert np.isfinite(res_huge.total_score)
        assert res_huge.total_score >= 95.0, "Áudio com ganho alto deve ser normalizado para identidade"

        res_tiny = shared_scorer.evaluate_imitation(BIRL_PATH, y_birl * 1e-10)
        assert np.isfinite(res_tiny.total_score)
        assert res_tiny.total_score < 30.0, "Áudio abaixo do limiar de energia deve ser tratado como inaudível"

    def test_extreme_durations(self, shared_scorer, birl_audio):
        """Áudios com duração ultra-curta (50ms) ou ultra-longa (10s) não devem quebrar o DTW."""
        y_birl, _ = birl_audio

        # 50 ms
        res_short = shared_scorer.evaluate_imitation(BIRL_PATH, y_birl[:800])
        assert np.isfinite(res_short.total_score)
        assert res_short.total_score < 25.0

        # ~10.4 s (12 repetições)
        res_long = shared_scorer.evaluate_imitation(BIRL_PATH, np.tile(y_birl, 12))
        assert np.isfinite(res_long.total_score)
        # DTW deve penalizar o descompasso de duração
        assert res_long.pitch_score < 20.0
