import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CLIPS_DIR = os.path.join(os.path.dirname(__file__), "data", "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

SR = 16000

def synthesize_vocal_phrase(
    duration: float,
    f0_contour: list[float],
    formants: list[float],
    energy_peaks: list[float],
    vowel_mod: bool = True
) -> np.ndarray:
    """
    Sintetizador formântico de fala que gera ondas acústicas ricas em harmônicos,
    modulações de pitch (F0) e envelopes de intensidade de fala realista.
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    
    f0_interp = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(f0_contour)), f0_contour)
    phase = 2 * np.pi * np.cumsum(f0_interp) / SR

    signal = np.zeros_like(t)
    for harmonic in range(1, 16):
        amp = (1.0 / harmonic)
        harmonic_freq = f0_interp * harmonic
        formant_boost = sum(np.exp(-((harmonic_freq - f)**2) / (2 * (100**2))) for f in formants)
        signal += amp * (1.0 + 1.8 * formant_boost) * np.sin(harmonic * phase)

    energy_interp = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(energy_peaks)), energy_peaks)
    signal = signal * energy_interp

    breath = np.random.randn(len(t)) * 0.03 * energy_interp
    signal = signal + breath

    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)

def generate_sample_clips():
    print("🎙️ Gerando arquivos de áudio de demonstração em 'backend/data/clips/'...")

    ref_file = os.path.join(CLIPS_DIR, "ta_saindo_da_jaula.wav")
    ref_duration = 5.9
    if os.path.exists(ref_file):
        try:
            ref_duration = round(sf.info(ref_file).duration, 2)
        except Exception:
            ref_duration = 5.9

    samples = [
        {
            "id": "ta_saindo_da_jaula",
            "title": "Tá Saindo da Jaula o Monstro",
            "phrase": "Tá saindo da jaula o monstro, porra!",
            "duration": ref_duration,
            "f0": [180, 240, 360, 420, 380, 290, 220, 180],
            "formants": [700, 1220, 2600],
            "energy": [0.2, 0.9, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1]
        },
        {
            "id": "pede_pra_sair",
            "title": "Tropa de Elite - Pede pra Sair",
            "phrase": "Pede pra sair! Pede pra sair, zero um!",
            "duration": 2.54,
            "f0": [160, 210, 230, 190, 170, 220, 200, 150],
            "formants": [800, 1300, 2700],
            "energy": [0.1, 0.8, 1.0, 0.7, 0.5, 0.9, 0.7, 0.1]
        },
        {
            "id": "ze_pequeno",
            "title": "Cidade de Deus - Zé Pequeno",
            "phrase": "Meu nome agora é Zé Pequeno, porra!",
            "duration": 3.16,
            "f0": [130, 135, 140, 165, 190, 175, 150, 130],
            "formants": [600, 1050, 2400],
            "energy": [0.2, 0.6, 0.7, 0.9, 1.0, 0.8, 0.5, 0.1]
        }
    ]

    meta_path = os.path.join(CLIPS_DIR, "metadata.json")

    for s in samples:
        file_path = os.path.join(CLIPS_DIR, f"{s['id']}.wav")
        if not os.path.exists(file_path):
            audio = synthesize_vocal_phrase(s["duration"], s["f0"], s["formants"], s["energy"])
            sf.write(file_path, audio, SR)
            print(f"  🔊 Criado: {s['id']}.wav ({s['title']})")
        else:
            print(f"  ℹ️ Já existente (preservado): {s['id']}.wav")

    if os.path.exists(ref_file):
        from backend.ml.dataset import augment_positive
        y_ref, _ = librosa.load(ref_file, sr=SR, mono=True)
        np.random.seed(42)
        good_attempt = augment_positive(y_ref)
    else:
        good_attempt = synthesize_vocal_phrase(
            duration=ref_duration,
            f0_contour=[185, 245, 355, 415, 375, 285, 215, 175],
            formants=[710, 1230, 2580],
            energy_peaks=[0.2, 0.88, 0.98, 0.89, 0.68, 0.38, 0.22, 0.1]
        )
    sf.write(os.path.join(CLIPS_DIR, "test_attempt_good.wav"), good_attempt, SR)
    print("  ⭐ Criado: test_attempt_good.wav (imitação próxima de 'ta_saindo_da_jaula')")

    t = np.linspace(0, ref_duration, int(SR * ref_duration), endpoint=False)
    bad_attempt = (0.25 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(os.path.join(CLIPS_DIR, "test_attempt_bad.wav"), bad_attempt, SR)
    print("  ❌ Criado: test_attempt_bad.wav (imitação monótona/desafinada)")

    metadata = []
    for s in samples:
        metadata.append({
            "id": s["id"],
            "title": s["title"],
            "phrase": s["phrase"],
            "file": f"{s['id']}.wav",
            "duration": s["duration"]
        })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("📄 'metadata.json' atualizado com os 3 memes oficiais!")


if __name__ == "__main__":
    generate_sample_clips()
