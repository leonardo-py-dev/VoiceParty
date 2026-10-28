"""
tests/test_e2e_suite.py
Suíte de Testes Ponta a Ponta Opaque-Box (Tiers 2, 3 e 4)
VoiceParty AI Core

Cobre:
- Tier 2: Boundary & Corner Cases (áudios ultracurtos, longos, silêncio, ruído, taxas de amostragem, arquivos corrompidos)
- Tier 3: Cross-Feature Combinations (upload + compare, integridade de streaming, divergência entre referências, ponderação de texto)
- Tier 4: Real-World Application Scenarios (jornada completa do jogador, playground arbitrário, separabilidade estatística, latência de inferência < 1.5s)
"""

import io
import os
import time
import wave
import json
import shutil
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, METADATA_PATH, CLIPS_DIR


# ============================================================================
# Helpers e Fixtures
# ============================================================================

def make_wav_bytes(
    duration_s: float = 1.0,
    freq: float = 440.0,
    sr: int = 16000,
    channels: int = 1,
    silence: bool = False,
    noise: bool = False
) -> bytes:
    """Gera dados WAV PCM 16-bit com configurações personalizadas para testes de estresse acústico."""
    num_samples = int(duration_s * sr)
    
    if silence:
        data = np.zeros((num_samples, channels), dtype=np.int16)
    elif noise:
        data = (np.random.normal(0, 0.2, (num_samples, channels)) * 32767).clip(-32767, 32767).astype(np.int16)
    else:
        t = np.linspace(0, duration_s, num_samples, endpoint=False)
        single_ch = 0.5 * np.sin(2 * np.pi * freq * t)
        if channels == 1:
            data = (single_ch * 32767).astype(np.int16).reshape(-1, 1)
        else:
            # Estéreo com fases ligeiramente distintas
            ch2 = 0.5 * np.cos(2 * np.pi * freq * t)
            data = np.stack([single_ch, ch2], axis=1)
            data = (data * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """Cliente HTTP in-memory para chamadas FastAPI."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def managed_catalog():
    """Fixture de isolamento que restaura o catálogo e remove clipes temporários."""
    backup_metadata = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            backup_metadata = f.read()

    files_to_cleanup = []

    yield files_to_cleanup

    for fpath in files_to_cleanup:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass

    if backup_metadata is not None:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            f.write(backup_metadata)


# ============================================================================
# TIER 2: Boundary & Corner Cases
# ============================================================================

def test_boundary_ultrashort_audio(client):
    """[Tier 2.1] Áudio ultracurto (80ms): o pipeline deve processar sem erro de indexação de frames."""
    short_wav = make_wav_bytes(duration_s=0.08, freq=440.0)
    files = {"user_file": ("ultrashort.wav", short_wav, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200, f"Falha com áudio ultracurto: {response.text}"
    result = response.json()
    assert 0.0 <= result["total_score"] <= 100.0


def test_boundary_long_audio(client):
    """[Tier 2.2] Áudio de longa duração (14s): o sistema deve processar e reamostrar sem estourar memória."""
    long_wav = make_wav_bytes(duration_s=14.0, freq=300.0)
    files = {"user_file": ("long.wav", long_wav, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200, f"Falha com áudio longo: {response.text}"
    result = response.json()
    assert 0.0 <= result["total_score"] <= 100.0


def test_boundary_pure_silence_no_crash(client):
    """[Tier 2.3] Silêncio absoluto (vetor de zeros): não deve ocorrer ZeroDivisionError no cálculo de z-score ou RMS."""
    silence_wav = make_wav_bytes(duration_s=1.5, silence=True)
    files = {"user_file": ("silence.wav", silence_wav, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200, f"Falha com silêncio puro: {response.text}"
    result = response.json()
    
    # Silêncio contra 'ta_saindo_da_jaula' não deve pontuar como imitação perfeita
    assert result["total_score"] < 70.0, f"Silêncio pontuou excessivamente alto: {result['total_score']}"
    assert 0.0 <= result["total_score"] <= 100.0


def test_boundary_pure_noise_discrimination(client):
    """[Tier 2.4] Ruído gaussiano (estática pura): o modelo não deve classificar como imitação de alto padrão."""
    noise_wav = make_wav_bytes(duration_s=1.2, noise=True)
    files = {"user_file": ("noise.wav", noise_wav, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200
    result = response.json()
    
    # Ruído aleatório não deve pontuar como imitação excelente (total_score < 90)
    assert result["total_score"] < 90.0, f"Ruído pontuou total excessivamente alto: {result['total_score']}"
    assert 0.0 <= result["total_score"] <= 100.0


def test_boundary_different_sampling_rates(client):
    """[Tier 2.5] Taxa de amostragem não-padrão (22050 Hz e 44100 Hz): o pipeline acústico deve resamplear sem falhas."""
    wav_22k = make_wav_bytes(duration_s=1.0, freq=350.0, sr=22050)
    files = {"user_file": ("audio_22k.wav", wav_22k, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    res_22k = client.post("/api/compare", data=data, files=files)
    assert res_22k.status_code == 200, f"Falha com áudio 22.05kHz: {res_22k.text}"

    wav_44k = make_wav_bytes(duration_s=1.0, freq=350.0, sr=44100)
    files_44k = {"user_file": ("audio_44k.wav", wav_44k, "audio/wav")}
    res_44k = client.post("/api/compare", data=data, files=files_44k)
    assert res_44k.status_code == 200, f"Falha com áudio 44.1kHz: {res_44k.text}"


def test_boundary_stereo_audio(client):
    """[Tier 2.6] Áudio com 2 canais (estéreo): o conversor deve tratar e converter em mono sem erro."""
    stereo_wav = make_wav_bytes(duration_s=1.0, freq=400.0, channels=2)
    files = {"user_file": ("stereo.wav", stereo_wav, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200, f"Falha ao processar áudio estéreo: {response.text}"
    assert 0.0 <= response.json()["total_score"] <= 100.0


def test_boundary_corrupted_audio_file(client):
    """[Tier 2.7] Envio de dados corrompidos (não-áudio com extensão .wav):
    O servidor deve tratar com erro HTTP 400/422/500 ou fallback de silêncio com nota rebaixada sem crash.
    """
    garbage_bytes = b"ESTE_ARQUIVO_NAO_E_UM_AUDIO_VALIDO_1234567890" * 50
    files = {"user_file": ("corrompido.wav", garbage_bytes, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    response = client.post("/api/compare", data=data, files=files)
    if response.status_code == 200:
        # Se entrar no fallback de silêncio de dataset.py, a pontuação não deve indicar boa imitação
        result = response.json()
        assert result["total_score"] < 70.0, f"Arquivo corrompido pontuou alto: {result['total_score']}"
    else:
        assert response.status_code in [400, 422, 500], f"Status code inesperado para arquivo inválido: {response.status_code}"


def test_boundary_upload_unicode_and_emojis(client, managed_catalog):
    """[Tier 2.8] Upload com caracteres UTF-8 complexos, emojis e acentuação no título."""
    audio_wav = make_wav_bytes(duration_s=0.6, freq=500.0)
    files = {"file": ("emoji_meme.wav", audio_wav, "audio/wav")}
    data = {
        "title": "🔥🎤 Meme Louco & Acentuação: Ação e Emoção! 日本語",
        "phrase": "Frase de teste com emojis 🚀"
    }

    response = client.post("/api/clips/upload", data=data, files=files)
    assert response.status_code == 200, f"Falha no upload com caracteres unicode: {response.text}"
    clip_info = response.json()["clip"]
    managed_catalog.append(os.path.join(CLIPS_DIR, clip_info["file"]))

    # O safe_id deve ser gerado sem quebrar codificação
    assert len(clip_info["id"]) > 0
    assert clip_info["title"] == data["title"]


# ============================================================================
# TIER 3: Cross-Feature Combinations
# ============================================================================

def test_cross_feature_upload_and_immediate_compare(client, managed_catalog):
    """[Tier 3.1] Upload de novo meme seguido de comparação imediata usando o novo ref_id gerado."""
    ref_audio = make_wav_bytes(duration_s=1.2, freq=440.0)
    user_attempt = make_wav_bytes(duration_s=1.2, freq=445.0)

    # 1. Faz upload do novo clipe
    upload_res = client.post(
        "/api/clips/upload",
        data={"title": "Meme Ciclo Completo", "phrase": "Testando pipeline integrado"},
        files={"file": ("ciclo_ref.wav", ref_audio, "audio/wav")}
    )
    assert upload_res.status_code == 200
    clip_id = upload_res.json()["clip"]["id"]
    managed_catalog.append(os.path.join(CLIPS_DIR, upload_res.json()["clip"]["file"]))

    # 2. Utiliza imediatamente o novo ref_id na comparação
    compare_res = client.post(
        "/api/compare",
        data={"ref_id": clip_id},
        files={"user_file": ("attempt.wav", user_attempt, "audio/wav")}
    )
    assert compare_res.status_code == 200, f"Falha ao comparar com clipe recém-criado: {compare_res.text}"
    result = compare_res.json()
    assert 0.0 <= result["total_score"] <= 100.0


def test_cross_feature_upload_and_audio_streaming_integrity(client, managed_catalog):
    """[Tier 3.2] Upload de clipe seguido de requisição ao endpoint de streaming para validar integridade binária."""
    original_audio = make_wav_bytes(duration_s=0.7, freq=600.0)
    files = {"file": ("streaming_check.wav", original_audio, "audio/wav")}
    data = {"title": "Teste Streaming Integridade", "phrase": "Audio original deve bater byte a byte"}

    upload_res = client.post("/api/clips/upload", data=data, files=files)
    assert upload_res.status_code == 200
    saved_filename = upload_res.json()["clip"]["file"]
    managed_catalog.append(os.path.join(CLIPS_DIR, saved_filename))

    # Baixa o áudio via endpoint de streaming
    audio_res = client.get(f"/api/clips/audio/{saved_filename}")
    assert audio_res.status_code == 200
    assert audio_res.content == original_audio, "O áudio servido via streaming diverge dos bytes enviados no upload"


def test_cross_feature_comparison_divergence_across_references(client):
    """[Tier 3.3] Compara uma mesma tentativa contra duas referências contrastantes comprovando sensibilidade acústica."""
    user_audio = make_wav_bytes(duration_s=1.0, freq=300.0)
    files_1 = {"user_file": ("attempt.wav", user_audio, "audio/wav")}
    files_2 = {"user_file": ("attempt.wav", user_audio, "audio/wav")}

    res_birl = client.post("/api/compare", data={"ref_id": "ta_saindo_da_jaula"}, files=files_1).json()
    res_jessica = client.post("/api/compare", data={"ref_id": "pede_pra_sair"}, files=files_2).json()

    # Como as referências são distintas, a pontuação não deve ser idêntica
    assert res_birl["total_score"] != res_jessica["total_score"] or res_birl["pitch_score"] != res_jessica["pitch_score"], \
        "Pontuações contra referências completamente diferentes não deveriam ser idênticas"


def test_cross_feature_text_weighting_toggle(client):
    """[Tier 3.4] Verifica a ativação e impacto da ponderação textual quando target_phrase e user_phrase são passados."""
    user_audio = make_wav_bytes(duration_s=1.0, freq=440.0)
    
    # 1. Sem texto
    res_no_text = client.post(
        "/api/compare",
        data={"ref_id": "ta_saindo_da_jaula"},
        files={"user_file": ("attempt.wav", user_audio, "audio/wav")}
    ).json()
    assert res_no_text.get("word_score") is None

    # 2. Com texto idêntico (100% de similaridade de palavras)
    res_match_text = client.post(
        "/api/compare",
        data={
            "ref_id": "ta_saindo_da_jaula",
            "target_phrase": "Birl! Não vai dá não!",
            "user_phrase": "Birl! Não vai dá não!"
        },
        files={"user_file": ("attempt.wav", user_audio, "audio/wav")}
    ).json()
    assert res_match_text.get("word_score") == 100.0

    # 3. Com texto completamente errado (0% ou baixo)
    res_wrong_text = client.post(
        "/api/compare",
        data={
            "ref_id": "ta_saindo_da_jaula",
            "target_phrase": "Birl! Não vai dá não!",
            "user_phrase": "Batatinha quando nasce esparrama pelo chão"
        },
        files={"user_file": ("attempt.wav", user_audio, "audio/wav")}
    ).json()
    assert res_wrong_text.get("word_score") < 35.0

    # A nota com texto idêntico deve ser estritamente maior que com texto errado
    assert res_match_text["total_score"] > res_wrong_text["total_score"]


# ============================================================================
# TIER 4: Real-World Application Scenarios
# ============================================================================

def test_scenario_full_player_journey(client):
    """[Tier 4.1] Jornada completa do usuário no jogo:
    1. Consulta /api/health para verificar prontidão da IA
    2. Consulta /api/clips para listar desafios
    3. Baixa áudio de referência escolhido via /api/clips/audio/{file}
    4. Envia tentativa via /api/compare
    5. Recebe feedback explicativo, pontuação e curvas para visualização gráfica
    """
    # Passo 1: Health
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "online"

    # Passo 2: Catálogo
    clips_res = client.get("/api/clips")
    assert clips_res.status_code == 200
    catalog = clips_res.json()
    chosen_clip = catalog[0]  # Escolhe primeiro meme disponível

    # Passo 3: Áudio de referência
    audio_res = client.get(f"/api/clips/audio/{chosen_clip['file']}")
    assert audio_res.status_code == 200
    assert len(audio_res.content) > 0

    # Passo 4: Submissão da tentativa
    user_attempt = make_wav_bytes(duration_s=1.0, freq=300.0)
    files = {"user_file": ("attempt.wav", user_attempt, "audio/wav")}
    data = {
        "ref_id": chosen_clip["id"],
        "target_phrase": chosen_clip["phrase"],
        "user_phrase": chosen_clip["phrase"]
    }
    compare_res = client.post("/api/compare", data=data, files=files)
    assert compare_res.status_code == 200
    result = compare_res.json()

    # Passo 5: Diagnóstico e Visualizações
    assert "total_score" in result
    assert "feedback" in result and len(result["feedback"]) > 10, "Feedback explicativo ausente ou curto demais"
    assert "visualizations" in result, "Visualizações ausentes na resposta"
    viz = result["visualizations"]
    assert "pitch_ref" in viz and "pitch_usr" in viz, "Curvas de pitch ausentes nas visualizações"
    assert len(viz["pitch_ref"]) > 0 and len(viz["pitch_usr"]) > 0


def test_scenario_interactive_playground_arbitrary_upload(client):
    """[Tier 4.2] Playground Interativo (R2): usuário fornece áudio de referência arbitrário e tentativa."""
    custom_ref = make_wav_bytes(duration_s=1.5, freq=440.0)
    custom_attempt = make_wav_bytes(duration_s=1.5, freq=435.0)

    files = {
        "ref_file": ("meu_audio_proprio.wav", custom_ref, "audio/wav"),
        "user_file": ("minha_tentativa.wav", custom_attempt, "audio/wav")
    }
    data = {
        "target_phrase": "Frase personalizada do usuário",
        "user_phrase": "Frase personalizada do usuário"
    }

    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200, f"Falha no fluxo do playground: {response.text}"
    body = response.json()
    assert 0.0 <= body["total_score"] <= 100.0
    assert body.get("word_score") == 100.0


def test_scenario_statistical_separability_good_vs_bad(client):
    """[Tier 4.3] Validação de Separabilidade Acústica (Acceptance Criteria R1):
    Avalia os áudios reais de teste ('test_attempt_good.wav' vs 'test_attempt_bad.wav')
    contra a referência original ('ta_saindo_da_jaula.wav').
    A imitação boa deve obter pontuação total superior e separabilidade neural
    estatisticamente significativa (Delta Neural > 20 pontos comprovado).
    """
    good_path = os.path.join(CLIPS_DIR, "test_attempt_good.wav")
    bad_path = os.path.join(CLIPS_DIR, "test_attempt_bad.wav")
    
    assert os.path.exists(good_path), f"Arquivo não encontrado: {good_path}"
    assert os.path.exists(bad_path), f"Arquivo não encontrado: {bad_path}"

    with open(good_path, "rb") as f_good:
        files_good = {"user_file": ("test_attempt_good.wav", f_good.read(), "audio/wav")}
    with open(bad_path, "rb") as f_bad:
        files_bad = {"user_file": ("test_attempt_bad.wav", f_bad.read(), "audio/wav")}

    data = {"ref_id": "ta_saindo_da_jaula"}

    res_good = client.post("/api/compare", data=data, files=files_good).json()
    res_bad = client.post("/api/compare", data=data, files=files_bad).json()

    score_good = res_good["total_score"]
    score_bad = res_bad["total_score"]
    neural_good = res_good["neural_score"]
    neural_bad = res_bad["neural_score"]

    print(f"\n[Separabilidade] Boa: {score_good} (Neural: {neural_good}) | Ruim: {score_bad} (Neural: {neural_bad})")

    # A boa imitação DEVE ser estatisticamente superior à ruim
    assert score_good > score_bad, f"Boa imitação ({score_good}) não superou a má imitação ({score_bad})"
    assert neural_good > neural_bad, f"Similaridade neural da boa ({neural_good}) não superou a má ({neural_bad})"
    
    # Delta neural demonstrado da rede siamesa (superior a 20 pontos de separabilidade)
    delta_neural = neural_good - neural_bad
    assert delta_neural >= 20.0, f"Delta neural de separabilidade ({delta_neural:.1f}) inferior a 20 pontos"


def test_scenario_inference_latency_cpu_under_threshold(client):
    """[Tier 4.4] Validação de Latência em CPU (Acceptance Criteria R1):
    A inferência completa de ponta a ponta por tentativa deve ocorrer em menos de 1,5 segundos (1500 ms).
    """
    good_path = os.path.join(CLIPS_DIR, "test_attempt_good.wav")
    with open(good_path, "rb") as f:
        audio_bytes = f.read()

    files = {"user_file": ("test_attempt_good.wav", audio_bytes, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}

    # Executa medição de tempo ponta a ponta
    start_time = time.perf_counter()
    response = client.post("/api/compare", data=data, files=files)
    elapsed_time = time.perf_counter() - start_time

    assert response.status_code == 200
    print(f"\n[Latência CPU] Tempo decorrido de inferência: {elapsed_time:.3f} segundos")

    # Limiar do Acceptance Criteria: < 1.5s em CPU local
    assert elapsed_time < 1.5, f"Latência de inferência ({elapsed_time:.3f}s) excedeu o limite máximo de 1.5s"
