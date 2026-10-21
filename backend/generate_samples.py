import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
    
    # Interpolação da curva de pitch
    f0_interp = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(f0_contour)), f0_contour)
    phase = 2 * np.pi * np.cumsum(f0_interp) / SR

    # Geração de harmônicos vocais (cordas vocais)
    signal = np.zeros_like(t)
    for harmonic in range(1, 16):
        # Decaimento natural de 6dB/oitava
        amp = (1.0 / harmonic)
        # Resonâncias formânticas da boca/garganta
        harmonic_freq = f0_interp * harmonic
        formant_boost = sum(np.exp(-((harmonic_freq - f)**2) / (2 * (100**2))) for f in formants)
        signal += amp * (1.0 + 1.8 * formant_boost) * np.sin(harmonic * phase)

    # Envelope de energia das sílabas
    energy_interp = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(energy_peaks)), energy_peaks)
    signal = signal * energy_interp

    # Leve ruído de fricativa/respiração
    breath = np.random.randn(len(t)) * 0.03 * energy_interp
    signal = signal + breath

    # Normalização
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)

def generate_sample_clips():
    print("🎙️ Gerando arquivos de áudio de demonstração em 'backend/data/clips/'...")

    samples = [
        {
            "id": "birl",
            "title": "Birl! O Monstro Tá Saindo da Jaula",
            "phrase": "Birl! Tá saindo da jaula o monstro!",
            "duration": 3.2,
            "f0": [140, 180, 240, 260, 210, 150, 170, 220, 180, 140],
            "formants": [700, 1220, 2600],
            "energy": [0.1, 0.9, 0.4, 0.8, 0.9, 0.3, 0.7, 0.9, 0.6, 0.1]
        },
        {
            "id": "ze_pequeno",
            "title": "Cidade de Deus - Zé Pequeno",
            "phrase": "Meu nome agora é Zé Pequeno!",
            "duration": 3.0,
            "f0": [130, 135, 140, 165, 190, 175, 150, 130],
            "formants": [600, 1050, 2400],
            "energy": [0.2, 0.6, 0.7, 0.9, 1.0, 0.8, 0.5, 0.1]
        },
        {
            "id": "pede_pra_sair",
            "title": "Tropa de Elite - Pede pra Sair",
            "phrase": "Pede pra sair, zero um!",
            "duration": 2.8,
            "f0": [160, 210, 230, 190, 170, 220, 200, 150],
            "formants": [800, 1300, 2700],
            "energy": [0.1, 0.8, 1.0, 0.7, 0.5, 0.9, 0.7, 0.1]
        },
        {
            "id": "ola_marilene",
            "title": "Olá Marilene",
            "phrase": "Olá Marilene, a noite é uma criança!",
            "duration": 3.5,
            "f0": [120, 150, 180, 140, 130, 145, 160, 135, 120],
            "formants": [500, 1500, 2500],
            "energy": [0.2, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.6, 0.1]
        }
    ]

    metadata = []
    for s in samples:
        audio = synthesize_vocal_phrase(s["duration"], s["f0"], s["formants"], s["energy"])
        file_path = os.path.join(CLIPS_DIR, f"{s['id']}.wav")
        sf.write(file_path, audio, SR)
        print(f"  🔊 Criado: {s['id']}.wav ({s['title']})")
        metadata.append({
            "id": s["id"],
            "title": s["title"],
            "phrase": s["phrase"],
            "file": f"{s['id']}.wav",
            "duration": s["duration"]
        })

    # Cria também duas tentativas simuladas do "birl": uma BOA imitação e uma RUIM para testes imediatos
    # Tentativa Boa: f0 e energia muito parecidas com BIRL
    good_birl = synthesize_vocal_phrase(
        duration=3.2,
        f0_contour=[145, 178, 235, 255, 205, 155, 172, 218, 175, 142],
        formants=[710, 1230, 2580],
        energy_peaks=[0.1, 0.88, 0.42, 0.78, 0.89, 0.32, 0.68, 0.88, 0.58, 0.1]
    )
    sf.write(os.path.join(CLIPS_DIR, "test_attempt_good.wav"), good_birl, SR)
    print("  ⭐ Criado: test_attempt_good.wav (imitação próxima de 'birl')")

    # Tentativa Ruim: fala monótona sem pitch e ritmo lento
    bad_birl = synthesize_vocal_phrase(
        duration=3.2,
        f0_contour=[110, 112, 110, 111, 109, 110, 112, 110, 109, 110],  # Robótico/Monótono
        formants=[400, 900, 2000],
        energy_peaks=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
    )
    sf.write(os.path.join(CLIPS_DIR, "test_attempt_bad.wav"), bad_birl, SR)
    print("  ❌ Criado: test_attempt_bad.wav (imitação monótona/desafinada)")

    with open(os.path.join(CLIPS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("📄 'metadata.json' atualizado com sucesso!")

if __name__ == "__main__":
    generate_sample_clips()
