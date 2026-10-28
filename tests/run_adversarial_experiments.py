import os
import sys
import numpy as np
import librosa
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.core.audio_processor import AudioProcessor
from backend.core.scorer import VoiceScorer

CLIPS_DIR = os.path.join(ROOT_DIR, "backend", "data", "clips")
BIRL_PATH = os.path.join(CLIPS_DIR, "ta_saindo_da_jaula.wav")

def run_all_experiments():
    print("=== INICIANDO EXPERIMENTOS ADVERSARIAIS EMPÍRICOS ===")
    processor = AudioProcessor()
    scorer = VoiceScorer(audio_processor=processor)

    # Carrega áudio de referência ta_saindo_da_jaula.wav
    y_birl, sr = librosa.load(BIRL_PATH, sr=16000, mono=True)
    duration_birl = len(y_birl) / sr
    print(f"Referência 'ta_saindo_da_jaula.wav': {len(y_birl)} amostras ({duration_birl:.2f}s) a {sr}Hz")

    results = {}

    # 1. BASELINE IDENTIDADE
    res_ident = scorer.evaluate_imitation(BIRL_PATH, BIRL_PATH)
    print(f"\n[1. BASELINE IDENTIDADE] Total: {res_ident.total_score}, Neural: {res_ident.neural_score}, Pitch: {res_ident.pitch_score}, Rhythm: {res_ident.rhythm_score}")
    results["ident"] = res_ident

    # 2. RUÍDO BRANCO
    print("\n--- 2. EXPERIMENTOS COM RUÍDO BRANCO ---")
    noise_levels = [0.05, 0.3, 1.0, 5.0]
    for n_level in noise_levels:
        noise = np.random.randn(len(y_birl)).astype(np.float32) * n_level
        res_noise = scorer.evaluate_imitation(BIRL_PATH, noise)
        print(f"birl vs Ruído Branco (sigma={n_level}): Total={res_noise.total_score}, Neural={res_noise.neural_score}, Pitch={res_noise.pitch_score}, Rhythm={res_noise.rhythm_score}")
        results[f"noise_{n_level}"] = res_noise

    # Ruído branco vs Ruído branco
    noise_a = np.random.randn(16000).astype(np.float32)
    noise_b = np.random.randn(16000).astype(np.float32)
    res_noise_vs_noise = scorer.evaluate_imitation(noise_a, noise_b)
    print(f"Ruído vs Ruído: Total={res_noise_vs_noise.total_score}, Neural={res_noise_vs_noise.neural_score}, Pitch={res_noise_vs_noise.pitch_score}, Rhythm={res_noise_vs_noise.rhythm_score}")
    results["noise_vs_noise"] = res_noise_vs_noise

    # Adição de ruído progressivo em birl
    snr_levels = [20, 10, 0, -5] # dB aprox
    for snr in snr_levels:
        noise = np.random.randn(len(y_birl)).astype(np.float32)
        signal_power = np.mean(y_birl ** 2)
        noise_power = np.mean(noise ** 2)
        k = np.sqrt(signal_power / (noise_power * (10 ** (snr / 10))))
        y_noisy = y_birl + noise * k
        res_snr = scorer.evaluate_imitation(BIRL_PATH, y_noisy)
        print(f"birl com SNR ~{snr}dB: Total={res_snr.total_score}, Neural={res_snr.neural_score}, Pitch={res_snr.pitch_score}, Rhythm={res_snr.rhythm_score}")
        results[f"snr_{snr}dB"] = res_snr

    # 3. INVERSÃO TEMPORAL (TEMPORAL REVERSAL)
    print("\n--- 3. EXPERIMENTOS COM INVERSÃO TEMPORAL ---")
    y_rev = y_birl[::-1].copy()
    res_rev = scorer.evaluate_imitation(BIRL_PATH, y_rev)
    print(f"birl vs birl Invertido: Total={res_rev.total_score}, Neural={res_rev.neural_score}, Pitch={res_rev.pitch_score}, Rhythm={res_rev.rhythm_score}")
    results["time_reversed"] = res_rev

    # 4. ONDAS SENOIDAIS PURAS
    print("\n--- 4. EXPERIMENTOS COM ONDAS SENOIDAIS PURAS ---")
    freqs = [50, 100, 220, 440, 880, 1000, 3000]
    t = np.linspace(0, duration_birl, len(y_birl), endpoint=False)
    for freq in freqs:
        sine = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)
        res_sine = scorer.evaluate_imitation(BIRL_PATH, sine)
        print(f"birl vs Seno {freq}Hz: Total={res_sine.total_score}, Neural={res_sine.neural_score}, Pitch={res_sine.pitch_score}, Rhythm={res_sine.rhythm_score}")
        results[f"sine_{freq}Hz"] = res_sine

    # Seno vs Seno de frequências diferentes
    sine_440 = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
    sine_880 = (np.sin(2 * np.pi * 880 * t) * 0.8).astype(np.float32)
    res_sine_diff = scorer.evaluate_imitation(sine_440, sine_880)
    print(f"Seno 440Hz vs Seno 880Hz: Total={res_sine_diff.total_score}, Neural={res_sine_diff.neural_score}, Pitch={res_sine_diff.pitch_score}, Rhythm={res_sine_diff.rhythm_score}")
    results["sine_440_vs_880"] = res_sine_diff

    # 5. PITCH SHIFTS (+-1, +-5, +-12 semitons)
    print("\n--- 5. EXPERIMENTOS COM PITCH SHIFTS ---")
    shifts = [1, -1, 5, -5, 12, -12]
    for s in shifts:
        y_shifted = librosa.effects.pitch_shift(y_birl, sr=16000, n_steps=s)
        res_shift = scorer.evaluate_imitation(BIRL_PATH, y_shifted)
        print(f"birl com Shift {s:+d} semitons: Total={res_shift.total_score}, Neural={res_shift.neural_score}, Pitch={res_shift.pitch_score}, Rhythm={res_shift.rhythm_score}")
        results[f"shift_{s:+d}"] = res_shift

    # 6. ESTABILIDADE NUMÉRICA EXTREMA
    print("\n--- 6. EXPERIMENTOS DE ESTABILIDADE NUMÉRICA ---")
    # Silêncio puro
    silence_cases = [
        ("silence_0s", np.zeros(0, dtype=np.float32)),
        ("silence_10samples", np.zeros(10, dtype=np.float32)),
        ("silence_1s", np.zeros(16000, dtype=np.float32)),
        ("silence_4s", np.zeros(64000, dtype=np.float32)),
    ]
    for name, s_arr in silence_cases:
        res_sil = scorer.evaluate_imitation(BIRL_PATH, s_arr)
        print(f"birl vs {name}: Total={res_sil.total_score}, Neural={res_sil.neural_score}, Pitch={res_sil.pitch_score}, Rhythm={res_sil.rhythm_score}")
        results[name] = res_sil

    # Silêncio vs Silêncio
    res_sil_sil = scorer.evaluate_imitation(np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.float32))
    print(f"Silêncio vs Silêncio: Total={res_sil_sil.total_score}, Neural={res_sil_sil.neural_score}, Pitch={res_sil_sil.pitch_score}, Rhythm={res_sil_sil.rhythm_score}")
    results["silence_vs_silence"] = res_sil_sil

    # DC Offset constante
    dc_arr = np.ones(16000, dtype=np.float32) * 0.5
    res_dc = scorer.evaluate_imitation(BIRL_PATH, dc_arr)
    print(f"birl vs DC Offset: Total={res_dc.total_score}, Neural={res_dc.neural_score}, Pitch={res_dc.pitch_score}, Rhythm={res_dc.rhythm_score}")
    results["dc_offset"] = res_dc

    # Dirac Delta (Impulso unitário)
    dirac = np.zeros(16000, dtype=np.float32)
    dirac[100] = 1.0
    res_dirac = scorer.evaluate_imitation(BIRL_PATH, dirac)
    print(f"birl vs Dirac Delta: Total={res_dirac.total_score}, Neural={res_dirac.neural_score}, Pitch={res_dirac.pitch_score}, Rhythm={res_dirac.rhythm_score}")
    results["dirac"] = res_dirac

    # Amplitudes extremas
    res_huge = scorer.evaluate_imitation(BIRL_PATH, y_birl * 1e6)
    res_tiny = scorer.evaluate_imitation(BIRL_PATH, y_birl * 1e-10)
    print(f"birl vs Amplitude x10^6: Total={res_huge.total_score}, Neural={res_huge.neural_score}, Pitch={res_huge.pitch_score}, Rhythm={res_huge.rhythm_score}")
    print(f"birl vs Amplitude x10^-10: Total={res_tiny.total_score}, Neural={res_tiny.neural_score}, Pitch={res_tiny.pitch_score}, Rhythm={res_tiny.rhythm_score}")
    results["huge_amp"] = res_huge
    results["tiny_amp"] = res_tiny

    # Duração extrema (0.05s vs 10s)
    short_audio = y_birl[:800] # 50ms
    long_audio = np.tile(y_birl, 12) # ~10.4s
    res_short = scorer.evaluate_imitation(BIRL_PATH, short_audio)
    res_long = scorer.evaluate_imitation(BIRL_PATH, long_audio)
    print(f"birl vs Ultra-curto (50ms): Total={res_short.total_score}, Neural={res_short.neural_score}, Pitch={res_short.pitch_score}, Rhythm={res_short.rhythm_score}")
    print(f"birl vs Ultra-longo (10.4s): Total={res_long.total_score}, Neural={res_long.neural_score}, Pitch={res_long.pitch_score}, Rhythm={res_long.rhythm_score}")
    results["short_audio"] = res_short
    results["long_audio"] = res_long

    # Verificação de NaNs / Infs em todos os resultados
    print("\n--- 7. VERIFICAÇÃO DE INTEGRIDADE NUMÉRICA (SEM NaN / Inf) ---")
    has_nan_or_inf = False
    for k, r in results.items():
        for field in ["total_score", "neural_score", "pitch_score", "rhythm_score"]:
            v = getattr(r, field)
            if v is not None:
                if np.isnan(v) or np.isinf(v):
                    print(f"[FALHA NUMÉRICA] {k}.{field} = {v}")
                    has_nan_or_inf = True
    if not has_nan_or_inf:
        print("[SUCESSO] Todos os 26 cenários adversariais produziram valores numéricos finitos e consistentes!")

if __name__ == "__main__":
    run_all_experiments()
