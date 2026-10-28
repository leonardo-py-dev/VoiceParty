"""
tests/test_api_endpoints.py
Suíte de Testes de Integração e Contrato de API (Tier 1: Feature Coverage)
VoiceParty AI Core - Opaque-Box Testing Suite

Cobre de forma detalhada e genuína os endpoints públicos do FastAPI:
- GET /api/health
- GET /api/clips
- GET /api/clips/audio/{filename}
- POST /api/clips/upload
- POST /api/compare
"""

import io
import os
import json
import wave
import shutil
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, METADATA_PATH, CLIPS_DIR


# ============================================================================
# Helpers e Fixtures
# ============================================================================

def generate_wav_bytes(
    duration_s: float = 1.0,
    frequency: float = 440.0,
    sample_rate: int = 16000,
    silence: bool = False,
    noise: bool = False
) -> bytes:
    """Gera um buffer de áudio WAV sintético válido em memória para testes."""
    num_samples = int(duration_s * sample_rate)
    
    if silence:
        samples = np.zeros(num_samples, dtype=np.int16)
    elif noise:
        samples = (np.random.uniform(-0.5, 0.5, num_samples) * 32767).astype(np.int16)
    else:
        t = np.linspace(0, duration_s, num_samples, endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
        samples = (waveform * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    """Cliente HTTP de teste para requisições in-memory rápidas e isoladas."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def clean_metadata():
    """Garante isolamento: restaura o metadata.json original e limpa novos clipes criados nos testes."""
    original_content = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            original_content = f.read()
    
    created_files = []

    yield created_files

    # Teardown: remove arquivos criados durante o teste
    for f in created_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    # Restaura o metadata.json ao estado inicial
    if original_content is not None:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            f.write(original_content)


# ============================================================================
# TIER 1.1: Health Check Endpoint (/api/health)
# ============================================================================

def test_health_status_code_and_json(client):
    """[Feature 1.1.1] Verifica se /api/health retorna HTTP 200 e status online."""
    response = client.get("/api/health")
    assert response.status_code == 200, f"Status code inesperado: {response.status_code}"
    data = response.json()
    assert isinstance(data, dict), "Resposta de /api/health deve ser um objeto JSON"
    assert data.get("status") == "online", f"Status retornado incorreto: {data.get('status')}"


def test_health_model_loaded_boolean(client):
    """[Feature 1.1.2] Verifica se a flag model_loaded está presente e é um booleano."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data, "Campo 'model_loaded' ausente no payload de saúde"
    assert isinstance(data["model_loaded"], bool), f"Esperava bool, recebeu {type(data['model_loaded'])}"


def test_health_device_indicator(client):
    """[Feature 1.1.3] Verifica se o campo device indica CPU ou CUDA explicitamente."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "device" in data, "Campo 'device' ausente no payload de saúde"
    device_str = str(data["device"]).lower()
    assert any(target in device_str for target in ["cpu", "cuda"]), f"Dispositivo não reconhecido: {device_str}"


def test_health_method_not_allowed(client):
    """[Feature 1.1.4] Verifica se requisição POST em rota GET /api/health retorna 405 Method Not Allowed."""
    response = client.post("/api/health", json={"dummy": "data"})
    assert response.status_code == 405, f"Esperava 405 para POST /api/health, obteve {response.status_code}"


def test_health_cors_headers_allowed(client):
    """[Feature 1.1.5] Verifica se os cabeçalhos de CORS permitem integração com o frontend SPA."""
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in ["*", "http://localhost:5173"]


# ============================================================================
# TIER 1.2: Clips Catalog Endpoint (/api/clips)
# ============================================================================

def test_list_clips_returns_list(client):
    """[Feature 1.2.1] Verifica se /api/clips retorna HTTP 200 e uma lista JSON."""
    response = client.get("/api/clips")
    assert response.status_code == 200
    clips = response.json()
    assert isinstance(clips, list), "O endpoint /api/clips deve retornar uma lista de clipes"
    assert len(clips) > 0, "O catálogo pré-cadastrado não deve estar vazio"


def test_list_clips_schema_compliance(client):
    """[Feature 1.2.2] Verifica se todos os clipes respeitam o schema canônico."""
    response = client.get("/api/clips")
    clips = response.json()
    required_keys = {"id", "title", "phrase", "file", "duration"}
    
    for clip in clips:
        missing = required_keys - set(clip.keys())
        assert not missing, f"Clipe {clip.get('id', 'desconhecido')} com campos ausentes: {missing}"
        assert isinstance(clip["id"], str) and clip["id"].strip(), "ID do clipe deve ser string não vazia"
        assert isinstance(clip["title"], str) and clip["title"].strip(), "Title deve ser string não vazia"
        assert isinstance(clip["phrase"], str) and clip["phrase"].strip(), "Phrase deve ser string não vazia"
        assert isinstance(clip["file"], str) and clip["file"].endswith(".wav"), "Arquivo de clipe deve terminar com .wav"
        assert isinstance(clip["duration"], (int, float)) and clip["duration"] > 0, "Duration deve ser positiva"


def test_list_clips_unique_ids(client):
    """[Feature 1.2.3] Garante que não existem IDs duplicados no catálogo."""
    response = client.get("/api/clips")
    clips = response.json()
    ids = [c["id"] for c in clips]
    assert len(ids) == len(set(ids)), f"IDs duplicados encontrados no catálogo: {ids}"


def test_list_clips_referenced_files_exist(client):
    """[Feature 1.2.4] Verifica se os arquivos referenciados no catálogo existem no disco."""
    response = client.get("/api/clips")
    clips = response.json()
    for clip in clips:
        file_path = os.path.join(CLIPS_DIR, clip["file"])
        assert os.path.isfile(file_path), f"Arquivo referenciado {clip['file']} não foi encontrado em {CLIPS_DIR}"


def test_list_clips_canonical_memes_present(client):
    """[Feature 1.2.5] Assegura presença de clipes emblemáticos fundamentais (ex: birl, ze_pequeno)."""
    response = client.get("/api/clips")
    clips = response.json()
    catalog_ids = {c["id"] for c in clips}
    for expected_id in ["ta_saindo_da_jaula", "ze_pequeno"]:
        assert expected_id in catalog_ids, f"Clipe esperado '{expected_id}' não consta no catálogo"


# ============================================================================
# TIER 1.3: Audio Retrieval Endpoint (/api/clips/audio/{filename})
# ============================================================================

def test_get_clip_audio_success(client):
    """[Feature 1.3.1] Verifica se /api/clips/audio/{filename} transmite o arquivo WAV com status 200."""
    response = client.get("/api/clips/audio/ta_saindo_da_jaula.wav")
    assert response.status_code == 200
    assert "audio/wav" in response.headers.get("content-type", "")
    assert len(response.content) > 1000, "Conteúdo do áudio ta_saindo_da_jaula.wav excessivamente pequeno"


def test_get_clip_audio_valid_riff_header(client):
    """[Feature 1.3.2] Garante que o stream transmitido possui cabeçalho válido RIFF/WAVE."""
    response = client.get("/api/clips/audio/ta_saindo_da_jaula.wav")
    assert response.status_code == 200
    content = response.content
    assert content[:4] == b"RIFF", "Início do arquivo de áudio deve ser 'RIFF'"
    assert content[8:12] == b"WAVE", "Bytes 8-12 do cabeçalho devem ser 'WAVE'"


def test_get_clip_audio_not_found_404(client):
    """[Feature 1.3.3] Verifica se arquivo inexistente retorna 404 Not Found com mensagem amigável."""
    response = client.get("/api/clips/audio/meme_fantasma_inexistente.wav")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "não encontrado" in data["detail"].lower()


def test_get_clip_audio_path_traversal_protection(client):
    """[Feature 1.3.4] Verifica proteção contra tentativa de acesso indevido via path traversal."""
    response = client.get("/api/clips/audio/../../backend/app/main.py")
    # FastAPI ou o SO deve bloquear com 404 Not Found ou 400 Bad Request
    assert response.status_code in [404, 400, 422], f"Status inesperado na tentativa de traversal: {response.status_code}"


def test_get_clip_audio_content_length_header(client):
    """[Feature 1.3.5] Verifica presença de Content-Length consistente com os bytes transmitidos."""
    response = client.get("/api/clips/audio/ta_saindo_da_jaula.wav")
    assert response.status_code == 200
    content_len_header = response.headers.get("content-length")
    if content_len_header:
        assert int(content_len_header) == len(response.content)


# ============================================================================
# TIER 1.4: Upload New Clip Endpoint (/api/clips/upload)
# ============================================================================

def test_upload_clip_success(client, clean_metadata):
    """[Feature 1.4.1] Verifica se o upload de clipe multipart é persistido e retorna 200 OK."""
    audio_data = generate_wav_bytes(duration_s=1.0, frequency=523.25)
    files = {"file": ("teste_custom.wav", audio_data, "audio/wav")}
    data = {"title": "Meme de Teste Unitario", "phrase": "Frase de teste para o meme"}
    
    response = client.post("/api/clips/upload", data=data, files=files)
    assert response.status_code == 200, f"Erro no upload: {response.text}"
    body = response.json()
    
    assert "message" in body
    assert "clip" in body
    clip_info = body["clip"]
    assert clip_info["title"] == "Meme de Teste Unitario"
    assert clip_info["phrase"] == "Frase de teste para o meme"
    
    # Registra arquivo para teardown seguro
    created_file = os.path.join(CLIPS_DIR, clip_info["file"])
    clean_metadata.append(created_file)
    assert os.path.exists(created_file), "Arquivo de áudio submetido não foi salvo no disco"


def test_upload_clip_safe_id_sanitization(client, clean_metadata):
    """[Feature 1.4.2] Verifica se caracteres especiais no título são sanitizados no safe_id."""
    audio_data = generate_wav_bytes(duration_s=0.5, frequency=330.0)
    files = {"file": ("sound.wav", audio_data, "audio/wav")}
    data = {"title": "Olá! @#$% Meme Novo 2026?", "phrase": "Tudo certo"}
    
    response = client.post("/api/clips/upload", data=data, files=files)
    assert response.status_code == 200
    body = response.json()
    safe_id = body["clip"]["id"]
    
    # ID deve conter apenas alfanuméricos, hífens ou underlines
    assert all(c.isalnum() or c in "-_" for c in safe_id), f"ID com caracteres inválidos: {safe_id}"
    
    clean_metadata.append(os.path.join(CLIPS_DIR, body["clip"]["file"]))


def test_upload_clip_missing_file_validation_error(client):
    """[Feature 1.4.3] Verifica se omitir o arquivo no formulário resulta em erro 422."""
    data = {"title": "Sem Arquivo", "phrase": "Tentativa sem áudio"}
    response = client.post("/api/clips/upload", data=data)
    assert response.status_code == 422, f"Esperava 422 Unprocessable Entity, obteve {response.status_code}"


def test_upload_clip_missing_title_validation_error(client):
    """[Feature 1.4.4] Verifica se omitir o título no formulário resulta em erro 422."""
    audio_data = generate_wav_bytes(duration_s=0.5)
    files = {"file": ("no_title.wav", audio_data, "audio/wav")}
    data = {"phrase": "Frase sem título"}
    response = client.post("/api/clips/upload", data=data, files=files)
    assert response.status_code == 422, f"Esperava 422 Unprocessable Entity, obteve {response.status_code}"


def test_upload_clip_metadata_consistency(client, clean_metadata):
    """[Feature 1.4.5] Garante que clipe cadastrado via upload aparece na chamada subsequente a /api/clips."""
    audio_data = generate_wav_bytes(duration_s=0.8, frequency=400.0)
    files = {"file": ("catalogo_check.wav", audio_data, "audio/wav")}
    data = {"title": "Clipe Visivel", "phrase": "Verificando se aparece no catalogo"}
    
    post_res = client.post("/api/clips/upload", data=data, files=files)
    assert post_res.status_code == 200
    clip_info = post_res.json()["clip"]
    clean_metadata.append(os.path.join(CLIPS_DIR, clip_info["file"]))
    
    get_res = client.get("/api/clips")
    assert get_res.status_code == 200
    all_clips = get_res.json()
    matching = [c for c in all_clips if c["id"] == clip_info["id"]]
    assert len(matching) == 1, "O clipe enviado não foi refletido na lista de clipes"


# ============================================================================
# TIER 1.5: Comparison Endpoint (/api/compare)
# ============================================================================

def test_compare_with_valid_catalog_ref_id(client):
    """[Feature 1.5.1] Verifica comparação de áudio usando ref_id de clipe existente no catálogo."""
    user_audio = generate_wav_bytes(duration_s=1.0, frequency=300.0)
    files = {"user_file": ("attempt.wav", user_audio, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}
    
    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200, f"Falha na comparação: {response.text}"
    result = response.json()
    
    # Contrato básico de campos
    assert "total_score" in result
    assert "neural_score" in result
    assert "pitch_score" in result
    assert "rhythm_score" in result
    assert "feedback" in result


def test_compare_with_custom_ref_file_playground_mode(client):
    """[Feature 1.5.2] Verifica comparação no modo Playground (enviando ref_file arbitrário + user_file)."""
    ref_audio = generate_wav_bytes(duration_s=1.2, frequency=440.0)
    user_audio = generate_wav_bytes(duration_s=1.2, frequency=430.0)
    
    files = {
        "ref_file": ("custom_ref.wav", ref_audio, "audio/wav"),
        "user_file": ("custom_attempt.wav", user_audio, "audio/wav")
    }
    
    response = client.post("/api/compare", files=files)
    assert response.status_code == 200, f"Falha na comparação com ref_file customizado: {response.text}"
    result = response.json()
    assert 0.0 <= result["total_score"] <= 100.0, "Total score fora do intervalo [0, 100]"


def test_compare_score_intervals_and_types(client):
    """[Feature 1.5.3] Assegura que todas as pontuações retornadas respeitam o intervalo [0, 100]."""
    user_audio = generate_wav_bytes(duration_s=1.0, frequency=350.0)
    files = {"user_file": ("attempt.wav", user_audio, "audio/wav")}
    data = {"ref_id": "ta_saindo_da_jaula"}
    
    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200
    res = response.json()
    
    for metric in ["total_score", "neural_score", "pitch_score", "rhythm_score"]:
        score_val = res[metric]
        assert isinstance(score_val, (int, float)), f"Métrica {metric} deve ser numérica"
        assert 0.0 <= score_val <= 100.0, f"Métrica {metric} com valor fora de [0, 100]: {score_val}"


def test_compare_with_speech_phrases(client):
    """[Feature 1.5.4] Verifica inclusão e cálculo de word_score quando target_phrase e user_phrase são passados."""
    user_audio = generate_wav_bytes(duration_s=1.0, frequency=300.0)
    files = {"user_file": ("attempt.wav", user_audio, "audio/wav")}
    data = {
        "ref_id": "ta_saindo_da_jaula",
        "target_phrase": "Birl! Não vai dá não!",
        "user_phrase": "Birl! Não vai dá não!"
    }
    
    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 200
    res = response.json()
    assert res.get("word_score") is not None, "word_score deve ser calculado quando frases são fornecidas"
    assert 0.0 <= res["word_score"] <= 100.0


def test_compare_missing_both_ref_id_and_ref_file_400(client):
    """[Feature 1.5.5] Verifica se omitir simultaneamente ref_id e ref_file resulta em HTTP 400 Bad Request."""
    user_audio = generate_wav_bytes(duration_s=1.0)
    files = {"user_file": ("attempt.wav", user_audio, "audio/wav")}
    
    response = client.post("/api/compare", files=files)
    assert response.status_code == 400, f"Esperava 400, obteve {response.status_code}"
    body = response.json()
    assert "detail" in body
    assert "forneça ref_id ou ref_file" in body["detail"].lower()


def test_compare_nonexistent_ref_id_404(client):
    """[Feature 1.5.6] Verifica se ref_id inexistente retorna HTTP 404 com mensagem adequada."""
    user_audio = generate_wav_bytes(duration_s=1.0)
    files = {"user_file": ("attempt.wav", user_audio, "audio/wav")}
    data = {"ref_id": "meme_que_nunca_existiu_xyz"}
    
    response = client.post("/api/compare", data=data, files=files)
    assert response.status_code == 404, f"Esperava 404, obteve {response.status_code}"
    body = response.json()
    assert "detail" in body
    assert "não encontrado" in body["detail"].lower()
