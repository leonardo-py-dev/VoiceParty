import os
import sys
import io
import time
import json
import uuid
import asyncio
import shutil
import tempfile
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import imageio_ffmpeg

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.core.scorer import VoiceScorer
from backend.ml.train import train_siamese_network
from backend.core.audio_processor import AudioProcessor
from backend.ml.dataset import (
    convert_to_wav,
    load_and_preprocess_audio,
    audio_to_log_mel,
    augment_positive,
    augment_negative,
    TARGET_SR,
    TARGET_SAMPLES,
)

CUSTOM_CLIPS_DIR = os.path.join(tempfile.gettempdir(), "voiceparty_custom_refs")
os.makedirs(CUSTOM_CLIPS_DIR, exist_ok=True)
CUSTOM_CLIPS: dict[str, dict] = {}

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI(title="VoiceParty Core AI Engine", version="1.0.0")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "clips"))
METADATA_PATH = os.path.join(CLIPS_DIR, "metadata.json")

# Instância compartilhada do avaliador com pré-aquecimento do catálogo
scorer = VoiceScorer()
scorer.processor.preload_catalog(CLIPS_DIR)

def validate_and_process_audio_bytes(raw_bytes: bytes, temp_dir: str, prefix: str = "audio") -> str:
    """
    Valida a integridade do arquivo de áudio recebido e padroniza para WAV 16kHz mono.
    Retorna explicitamente HTTP 400 Bad Request caso o arquivo esteja vazio, corrompido ou seja ilegível.
    """
    if not raw_bytes or len(raw_bytes) < 44:
        raise HTTPException(status_code=400, detail="Formato de áudio inválido ou corrompido")

    # 1. Tenta carregar diretamente com soundfile (leitura em memória ultrarrápida para WAV/FLAC/OGG)
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Formato de áudio inválido ou corrompido")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != TARGET_SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        dest_path = os.path.join(temp_dir, f"{prefix}_standard.wav")
        sf.write(dest_path, data, sr, subtype="PCM_16")
        return dest_path
    except HTTPException:
        raise
    except Exception:
        pass

    # 2. Se soundfile falhar (ex: WebM/Opus do MediaRecorder do Chrome), recorre ao ffmpeg
    raw_path = os.path.join(temp_dir, f"{prefix}_raw")
    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    converted_path = os.path.join(temp_dir, f"{prefix}_ffmpeg.wav")
    success = convert_to_wav(raw_path, converted_path, target_sr=TARGET_SR)
    if not success or not os.path.exists(converted_path):
        raise HTTPException(status_code=400, detail="Formato de áudio inválido ou corrompido")

    try:
        data, sr = sf.read(converted_path, dtype="float32")
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Formato de áudio inválido ou corrompido")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
            sf.write(converted_path, data, TARGET_SR, subtype="PCM_16")
        return converted_path
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Formato de áudio inválido ou corrompido")

def _build_structured_feedback(rank: str, result, scores: dict) -> dict:
    """Gera diagnóstico explicável, dicas acionáveis e estrutura compatível com PROJECT.md."""
    if rank == "S":
        title = "Atuação Lendária!"
        message = "🌟 Espetacular! Sua voz capturou com perfeição o timbre, afinação e ritmo do original."
        tips = [
            "Performance impecável!",
            "Mantenha o mesmo aquecimento vocal para os próximos memes."
        ]
    elif rank == "A":
        title = "Excelente Imitação!"
        message = "🔥 Muito perto do original! Seu timbre e cadência foram muito expressivos."
        tips = [
            "Tente alinhar com mais sutileza a respiração nas frases finais.",
            "Ajuste ligeiramente a entonação no pico da frase."
        ]
    elif rank == "B":
        title = "Boa Tentativa!"
        message = "👍 Dá para reconhecer claramente o meme, com bom ritmo e energia."
        tips = [
            "Pratique o ritmo das pausas para não adiantar a fala.",
            "Tente aproximar mais a colocação da voz do tom original."
        ]
    elif rank == "C":
        title = "Quase Lá!"
        message = "😅 Alguns trechos ficaram legais, mas a entonação e o ritmo oscilaram."
        tips = [
            "Escute o áudio de referência com fones antes de gravar.",
            "Concentre-se em reproduzir as subidas e descidas de tom."
        ]
    else:
        title = "Precisa Praticar!"
        message = "🎤 Ficou distante da referência original, mas não desista! A prática leva à perfeição."
        tips = [
            "Grave em um ambiente silencioso para capturar melhor a voz.",
            "Tente falar no mesmo ritmo e volume da referência original."
        ]

    # Se o scorer gerou feedback explicativo textual enriquecido, incorpora à mensagem
    if hasattr(result, "feedback") and isinstance(result.feedback, str) and len(result.feedback) > 0:
        message = result.feedback

    return {
        "title": title,
        "message": message,
        "tips": tips,
        "summary": message,
        "rank": rank,
        "pitch_diagnosis": "Afinação precisa" if scores["pitch"] >= 75 else "Oscilação de entonação",
        "rhythm_diagnosis": "Ritmo consistente" if scores["rhythm"] >= 75 else "Pausas fora do compasso",
        "neural_diagnosis": "Timbre similar" if scores["neural"] >= 75 else "Voz diferente da referência",
        "speech_diagnosis": "Pronúncia clara" if scores["speech"] >= 70 else "Palavras com baixa correspondência",
        "score": float(result.total_score),
        "overall": "Aprovado" if result.total_score >= 70 else "Em evolução"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": scorer.processor.is_model_loaded,
        "device": str(scorer.processor.device)
    }

@app.get("/api/clips")
def list_clips():
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/clips/audio/{filename}")
def get_clip_audio(filename: str):
    file_path = os.path.join(CLIPS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Áudio não encontrado")
    return FileResponse(file_path, media_type="audio/wav")

@app.post("/api/clips/upload")
async def upload_new_clip(
    title: str = Form(...),
    phrase: str = Form(...),
    file: UploadFile = File(...)
):
    """Permite ao usuário subir seus próprios memes ou falas de filmes."""
    safe_id = "".join(c for c in title.lower() if c.isalnum() or c in ('_', '-'))[:25]
    if not safe_id:
        safe_id = "meme_custom"
    
    filename = f"{safe_id}_{file.filename}"
    file_path = os.path.join(CLIPS_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Atualiza metadata.json
    clips = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            clips = json.load(f)

    new_entry = {
        "id": safe_id,
        "title": title,
        "phrase": phrase,
        "file": filename,
        "duration": 3.0
    }
    clips.append(new_entry)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)

    return {"message": "Áudio cadastrado com sucesso!", "clip": new_entry}

@app.post("/api/compare")
async def compare_audios(
    ref_id: Optional[str] = Form(None),
    ref_file: Optional[UploadFile] = File(None),
    user_file: UploadFile = File(...),
    target_phrase: Optional[str] = Form(None),
    user_phrase: Optional[str] = Form(None),
):
    """
    Endpoint principal do Marco 2: compara o áudio de referência com a gravação do usuário
    utilizando a Rede Neural Siamesa + Pitch DTW + Envelope RMS, retornando
    espectrogramas Log-Mel, curvas de afinação DTW, embeddings, radar e feedback estruturado.
    """
    t_start = time.perf_counter()
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Determina caminho do áudio de referência
        if ref_id:
            if ref_id in CUSTOM_CLIPS and os.path.exists(CUSTOM_CLIPS[ref_id]["path"]):
                ref_path = CUSTOM_CLIPS[ref_id]["path"]
            else:
                ref_path = os.path.join(CLIPS_DIR, f"{ref_id}.wav")
                if not os.path.exists(ref_path):
                    # Procura por nome do arquivo em metadata
                    clips = []
                    if os.path.exists(METADATA_PATH):
                        with open(METADATA_PATH, "r", encoding="utf-8") as f:
                            clips = json.load(f)
                    matched = next((c for c in clips if c["id"] == ref_id), None)
                    if matched:
                        ref_path = os.path.join(CLIPS_DIR, matched["file"])
                    else:
                        raise HTTPException(status_code=404, detail=f"Áudio de referência '{ref_id}' não encontrado")
                if not os.path.exists(ref_path):
                    raise HTTPException(status_code=404, detail=f"Áudio de referência '{ref_id}' não encontrado")
        elif ref_file:
            ref_bytes = await ref_file.read()
            ref_path = validate_and_process_audio_bytes(ref_bytes, temp_dir, prefix="custom_ref")
        else:
            raise HTTPException(status_code=400, detail="Forneça ref_id ou ref_file")

        # 2. Valida e padroniza o arquivo enviado pelo usuário (rejeita arquivos vazios ou corrompidos com HTTP 400)
        user_bytes = await user_file.read()
        user_wav_path = validate_and_process_audio_bytes(user_bytes, temp_dir, prefix="user_attempt")

        # 3. Executa a avaliação da IA (calcula similaridades e armazena features em cache)
        result = scorer.evaluate_imitation(
            ref_audio_path=ref_path,
            user_audio_path=user_wav_path,
            target_text=target_phrase,
            user_text=user_phrase
        )

        # 4. Recupera features acústicas e neurais em cache (instantâneo na RAM, sem recomputar)
        ref_feat = scorer.processor.get_features(ref_path)
        user_feat = scorer.processor.get_features(user_wav_path)

        # 5. Exporta matrizes e dados para visualização rica no frontend
        spectrogram_ref = scorer.processor.export_spectrogram(ref_feat["y"])
        spectrogram_usr = scorer.processor.export_spectrogram(user_feat["y"])
        mfcc_ref = scorer.processor.export_mfcc(ref_feat["y"])
        mfcc_usr = scorer.processor.export_mfcc(user_feat["y"])
        pitch_curve_data, pitch_ref_simple, pitch_usr_simple = scorer.processor.export_pitch_visualization(ref_feat, user_feat)
        embeddings_data = scorer.processor.export_embeddings_visualization(ref_feat, user_feat)
        radar_data = scorer.processor.export_radar_metrics(
            result.neural_score,
            result.pitch_score,
            result.rhythm_score,
            result.word_score,
            ref_feat,
            user_feat
        )

        # 6. Determina Rank e Scores estruturados
        total_score = float(result.total_score)
        if total_score >= 90.0:
            rank = "S"
        elif total_score >= 80.0:
            rank = "A"
        elif total_score >= 70.0:
            rank = "B"
        elif total_score >= 60.0:
            rank = "C"
        else:
            rank = "D"

        speech_score = float(result.word_score) if result.word_score is not None else float(radar_data["speech"])
        scores = {
            "neural": float(result.neural_score),
            "pitch": float(result.pitch_score),
            "rhythm": float(result.rhythm_score),
            "speech": speech_score
        }

        feedback_dict = _build_structured_feedback(rank, result, scores)
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "total_score": total_score,
            "rank": rank,
            "scores": scores,
            "feedback": feedback_dict,
            "visualizations": {
                "spectrogram_ref": spectrogram_ref,
                "spectrogram_usr": spectrogram_usr,
                "mfcc_ref": mfcc_ref,
                "mfcc_usr": mfcc_usr,
                "pitch_curve": pitch_curve_data,
                "embeddings": embeddings_data,
                "radar": radar_data,
                "pitch_ref": pitch_ref_simple,
                "pitch_usr": pitch_usr_simple
            },
            "latency_ms": latency_ms,
            # Campos legados para compatibilidade reversa com testes anteriores:
            "neural_score": float(result.neural_score),
            "pitch_score": float(result.pitch_score),
            "rhythm_score": float(result.rhythm_score),
            "word_score": result.word_score,
            "details": result.details
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/playground/custom-audio/{custom_id}")
def get_custom_audio(custom_id: str):
    entry = CUSTOM_CLIPS.get(custom_id)
    if not entry or not os.path.exists(entry["path"]):
        raise HTTPException(status_code=404, detail="Áudio personalizado não encontrado")
    return FileResponse(entry["path"], media_type="audio/wav")


@app.post("/api/playground/train-stream")
async def playground_train_stream(
    ref_file: UploadFile = File(...),
    title: Optional[str] = Form("Áudio Personalizado"),
    phrase: Optional[str] = Form(""),
    epochs: int = Form(10),
):
    """
    Processa o áudio enviado pelo usuário e executa o treinamento supervisionado da Rede Neural
    em tempo real, transmitindo cada etapa via NDJSON para visualização ao vivo no navegador.
    """
    raw_bytes = await ref_file.read()
    custom_id = f"custom_{uuid.uuid4().hex[:10]}"
    saved_wav_path = validate_and_process_audio_bytes(raw_bytes, CUSTOM_CLIPS_DIR, prefix=custom_id)
    epochs = max(4, min(int(epochs), 20))

    async def event_generator():
        device = scorer.processor.device
        model = scorer.processor.model

        # Etapa 1: Carregamento e Pré-processamento com Librosa
        y_raw, _ = librosa.load(saved_wav_path, sr=TARGET_SR, mono=True)
        y_trim, _ = librosa.effects.trim(y_raw, top_db=25)
        if len(y_trim) == 0:
            y_trim = y_raw if len(y_raw) > 0 else np.zeros(TARGET_SR, dtype=np.float32)
        max_v = float(np.max(np.abs(y_trim)))
        if max_v > 1e-6:
            y_trim = (y_trim / max_v).astype(np.float32)

        duration = round(float(len(y_trim) / TARGET_SR), 2)
        wave_idx = np.linspace(0, max(0, len(y_trim) - 1), 80).astype(int)
        waveform_preview = [round(float(y_trim[i]), 3) for i in wave_idx] if len(y_trim) > 0 else [0.0] * 80

        yield json.dumps({
            "stage": "preprocess",
            "message": f"Áudio padronizado via Librosa (16kHz Mono, silêncio removido, {duration}s)",
            "duration_sec": duration,
            "samples": int(len(y_trim)),
            "waveform": waveform_preview
        }, ensure_ascii=False) + "\n"
        await asyncio.sleep(0.35)

        # Etapa 2: Extração de MFCCs, Espectrograma Log-Mel, Pitch F0 e Energia RMS
        mfcc_matrix = scorer.processor.export_mfcc(y_trim, n_mfcc=13, target_frames=60)
        spec_matrix = scorer.processor.export_spectrogram(y_trim, n_mels=64, target_frames=60)
        f0_curve, voiced_mask = scorer.processor.extract_pitch_curve(y_trim, sr=TARGET_SR)
        pitch_60 = [
            round(float(v), 1)
            for v in np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, max(1, len(f0_curve))), f0_curve)
        ]
        rms_raw = librosa.feature.rms(y=y_trim, hop_length=256)[0]
        if np.max(rms_raw) > 0:
            rms_raw = rms_raw / np.max(rms_raw)
        rms_60 = [
            round(float(v), 3)
            for v in np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, max(1, len(rms_raw))), rms_raw)
        ]

        yield json.dumps({
            "stage": "features",
            "message": "Extraídos 13 MFCCs, Espectrograma Log-Mel (64 bandas), Curva de Pitch F0 e Envelope RMS",
            "mfcc": mfcc_matrix,
            "spectrogram": spec_matrix,
            "pitch_hz": pitch_60,
            "rms_env": rms_60
        }, ensure_ascii=False) + "\n"
        await asyncio.sleep(0.45)

        # Etapa 3: Construção do Dataset Supervisionado e Divisão Treino / Teste
        if len(y_trim) < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / max(1, len(y_trim))))
            y_anchor = np.tile(y_trim, repeats)[:TARGET_SAMPLES]
        else:
            y_anchor = y_trim[:TARGET_SAMPLES]

        catalog_waves = []
        for fname in ("ta_saindo_da_jaula.wav", "pede_pra_sair.wav", "ze_pequeno.wav"):
            fpath = os.path.join(CLIPS_DIR, fname)
            if os.path.exists(fpath):
                try:
                    catalog_waves.append(load_and_preprocess_audio(fpath))
                except Exception:
                    pass

        def build_triplets(n_triplets: int):
            anchors, positives, negatives = [], [], []
            for i in range(n_triplets):
                pos_w = augment_positive(y_anchor)
                if catalog_waves and (i % 2 == 0):
                    neg_w = augment_positive(catalog_waves[i % len(catalog_waves)])
                else:
                    neg_w = augment_negative(y_anchor)
                anchors.append(audio_to_log_mel(y_anchor))
                positives.append(audio_to_log_mel(pos_w))
                negatives.append(audio_to_log_mel(neg_w))
            return (
                torch.stack(anchors).to(device),
                torch.stack(positives).to(device),
                torch.stack(negatives).to(device),
            )

        train_a, train_p, train_n = build_triplets(16)
        val_a, val_p, val_n = build_triplets(8)

        yield json.dumps({
            "stage": "dataset",
            "message": "Dataset supervisionado separado: 16 tripletos de Treino (67%) e 8 de Teste/Validação (33%)",
            "train_samples": 16,
            "test_samples": 8,
            "total_samples": 24
        }, ensure_ascii=False) + "\n"
        await asyncio.sleep(0.4)

        # Etapa 4: Treinamento Supervisionado em Tempo Real (Triplet Margin Loss)
        # Congela extratores convolucionais de base e ajusta a cabeça de projeção com segurança
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.0006, weight_decay=1e-4)
        criterion = nn.TripletMarginLoss(margin=0.35, p=2.0)

        anchor_single = train_a[0:1]

        for ep in range(1, epochs + 1):
            optimizer.zero_grad()
            emb_a = model.forward_one(train_a)
            emb_p = model.forward_one(train_p)
            emb_n = model.forward_one(train_n)
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pos_dots_tr = torch.sum(emb_a * emb_p, dim=-1)
                neg_dots_tr = torch.sum(emb_a * emb_n, dim=-1)
                margin_tr = pos_dots_tr - neg_dots_tr
                # Acurácia de classificação de pares (Positivo > Negativo + margem progressiva)
                req_margin = max(0.05, 0.28 - ep * 0.015)
                correct_tr = (margin_tr > req_margin).float().mean().item()
                train_acc = round(min(100.0, float(correct_tr * 100.0)), 1)

                v_a = model.forward_one(val_a)
                v_p = model.forward_one(val_p)
                v_n = model.forward_one(val_n)
                v_loss = criterion(v_a, v_p, v_n).item()
                pos_dots_val = torch.sum(v_a * v_p, dim=-1)
                neg_dots_val = torch.sum(v_a * v_n, dim=-1)
                correct_val = ((pos_dots_val - neg_dots_val) > req_margin).float().mean().item()
                val_acc = round(min(100.0, float(correct_val * 100.0)), 1)

                cur_emb = model.forward_one(anchor_single).squeeze(0).cpu().numpy()

            yield json.dumps({
                "stage": "epoch",
                "epoch": ep,
                "total_epochs": epochs,
                "train_loss": round(float(loss.item()), 4),
                "val_loss": round(float(v_loss), 4),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "pos_similarity": round(float(pos_dots_tr.mean().item() * 100.0), 1),
                "neg_similarity": round(float(max(0.0, neg_dots_tr.mean().item()) * 100.0), 1),
                "embedding": [round(float(x), 4) for x in cur_emb],
                "message": f"Época {ep}/{epochs} — Loss: {loss.item():.4f} | Acurácia Treino: {train_acc}% | Acurácia Teste: {val_acc}%"
            }, ensure_ascii=False) + "\n"
            await asyncio.sleep(0.24)

        # Limpa o cache para recalcular embeddings com os pesos atualizados
        scorer.processor.clear_cache()
        scorer.processor.get_features(saved_wav_path)

        CUSTOM_CLIPS[custom_id] = {
            "id": custom_id,
            "title": title or "Áudio Personalizado",
            "phrase": phrase or "",
            "path": saved_wav_path,
            "duration": duration,
        }

        yield json.dumps({
            "stage": "complete",
            "custom_id": custom_id,
            "title": title or "Áudio Personalizado",
            "phrase": phrase or "",
            "duration_sec": duration,
            "audio_url": f"/api/playground/custom-audio/{custom_id}",
            "message": "✅ Treinamento concluído! A rede aprendeu a representação vocal do seu áudio. Agora grave sua tentativa abaixo para replicá-lo!"
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

STATIC_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

@app.get("/", response_class=HTMLResponse)
def main_ui():
    """Interface principal: desafios, gravacao, Playground e visualizacao passo a passo da IA."""
    return FileResponse(STATIC_INDEX, media_type="text/html")

@app.get("/legacy", response_class=HTMLResponse)
def diagnostic_ui():
    """Interface de teste para testar a rede neural, microfone e upload sem precisar de tela de jogo."""
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>VoiceParty — Bancada de Testes da IA</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-6 font-sans">
  <div class="max-w-4xl mx-auto space-y-6">
    
    <!-- Cabeçalho -->
    <header class="flex items-center justify-between border-b border-slate-800 pb-4">
      <div>
        <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
          🎤 VoiceParty AI Core
        </h1>
        <p class="text-sm text-slate-400 mt-1">Ambiente de teste e verificação da Rede Neural Siamesa e Comparador de Áudio</p>
      </div>
      <span class="px-3 py-1 bg-emerald-950 text-emerald-400 text-xs font-semibold rounded-full border border-emerald-800">
        IA Pronta & Treinada
      </span>
    </header>

    <!-- Área de Seleção e Gravação -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      
      <!-- Card Áudio Original -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 class="text-lg font-bold text-cyan-400 flex items-center gap-2">
          <span>1️⃣</span> Áudio Original (Referência)
        </h2>
        
        <div>
          <label class="block text-xs font-semibold text-slate-400 uppercase mb-2">Escolha um Meme / Fala:</label>
          <select id="clipSelect" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-cyan-500">
            <option value="">Carregando áudios...</option>
          </select>
        </div>

        <div id="clipPhraseBox" class="p-3 bg-slate-950 rounded-lg border border-slate-800 text-sm italic text-slate-300 min-h-[50px] flex items-center">
          Selecione um áudio acima...
        </div>

        <audio id="audioPlayer" controls class="w-full mt-2 h-10"></audio>
      </div>

      <!-- Card Tentativa do Usuário -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 class="text-lg font-bold text-purple-400 flex items-center gap-2">
          <span>2️⃣</span> Sua Tentativa (Microfone ou Arquivo)
        </h2>

        <div class="flex items-center gap-3">
          <button id="recordBtn" class="flex-1 py-3 px-4 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-lg transition-all flex items-center justify-center gap-2">
            <span id="recordIcon" class="w-3 h-3 bg-red-400 rounded-full"></span>
            <span id="recordText">Gravar Microfone</span>
          </button>
          <span id="timerText" class="text-sm font-mono text-slate-400">0.0s</span>
        </div>

        <audio id="userAudioPreview" controls class="w-full h-10 hidden"></audio>

        <div class="text-xs text-slate-400 text-center">ou selecione um arquivo de teste</div>
        <input type="file" id="fileUpload" accept="audio/*" class="w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700">
      </div>
    </div>

    <!-- Botão de Ação -->
    <div class="text-center pt-2">
      <button id="compareBtn" class="py-4 px-10 bg-gradient-to-r from-cyan-500 via-indigo-600 to-purple-600 hover:opacity-95 text-white font-extrabold text-lg rounded-xl shadow-lg shadow-indigo-950 transition-all transform active:scale-95 disabled:opacity-50">
        ⚡ Comparar Tentativa com a Rede Neural
      </button>
    </div>

    <!-- Resultados -->
    <div id="resultsCard" class="hidden bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
      <div class="flex flex-col md:flex-row items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Pontuação Geral</span>
          <div class="flex items-baseline gap-2 mt-1">
            <span id="totalScore" class="text-5xl font-black text-cyan-400">--</span>
            <span class="text-xl text-slate-500">/ 100</span>
          </div>
        </div>
        <div id="feedbackBox" class="p-4 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 max-w-xl">
          Aguardando análise...
        </div>
      </div>

      <!-- Barra de métricas -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        
        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800">
          <div class="text-xs text-slate-400 font-semibold mb-1">🧠 Rede Siamesa (CRNN)</div>
          <div id="neuralScore" class="text-2xl font-bold text-purple-400">--%</div>
          <p class="text-[11px] text-slate-500 mt-1">Similaridade acústica/timbre e expressão da voz</p>
        </div>

        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800">
          <div class="text-xs text-slate-400 font-semibold mb-1">🎵 Entonação (Pitch DTW)</div>
          <div id="pitchScore" class="text-2xl font-bold text-cyan-400">--%</div>
          <p class="text-[11px] text-slate-500 mt-1">Curva de subidas e descidas do tom vocal</p>
        </div>

        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800">
          <div class="text-xs text-slate-400 font-semibold mb-1">⏱️ Ritmo (RMS Dynamic)</div>
          <div id="rhythmScore" class="text-2xl font-bold text-emerald-400">--%</div>
          <p class="text-[11px] text-slate-500 mt-1">Tempo das pausas e intensidade das palavras</p>
        </div>
      </div>

      <!-- Gráfico de Comparação de Entonação -->
      <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
        <div class="flex items-center justify-between text-xs">
          <span class="font-bold text-slate-300">Curvas de Entonação Vocal (Pitch F0)</span>
          <div class="flex items-center gap-4">
            <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-cyan-400 rounded"></span> Original</span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-purple-400 rounded"></span> Sua Gravação</span>
          </div>
        </div>
        <canvas id="pitchCanvas" class="w-full h-36 bg-slate-900 rounded-lg"></canvas>
      </div>
    </div>

  </div>

  <script>
    let clips = [];
    let mediaRecorder = null;
    let audioChunks = [];
    let recordedBlob = null;
    let recordingInterval = null;
    let recordingSeconds = 0;

    // Carrega clips
    async function initClips() {
      const res = await fetch('/api/clips');
      clips = await res.json();
      const select = document.getElementById('clipSelect');
      select.innerHTML = '<option value="">-- Selecione um áudio --</option>';
      clips.forEach(c => {
        select.innerHTML += `<option value="${c.id}">${c.title}</option>`;
      });

      if (clips.length > 0) {
        select.value = clips[0].id;
        updateSelectedClip();
      }
    }

    function updateSelectedClip() {
      const selId = document.getElementById('clipSelect').value;
      const c = clips.find(x => x.id === selId);
      if (c) {
        document.getElementById('clipPhraseBox').innerText = `"${c.phrase}"`;
        document.getElementById('audioPlayer').src = `/api/clips/audio/${c.file}`;
      }
    }
    document.getElementById('clipSelect').addEventListener('change', updateSelectedClip);

    // Gravação via microfone
    const recordBtn = document.getElementById('recordBtn');
    const recordText = document.getElementById('recordText');
    const timerText = document.getElementById('timerText');
    const userAudioPreview = document.getElementById('userAudioPreview');

    recordBtn.addEventListener('click', async () => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        clearInterval(recordingInterval);
        recordBtn.className = "flex-1 py-3 px-4 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-lg transition-all flex items-center justify-center gap-2";
        recordText.innerText = "Gravar Novamente";
      } else {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
          const options = mimeType ? { mimeType } : {};
          mediaRecorder = new MediaRecorder(stream, options);
          audioChunks = [];
          recordingSeconds = 0;
          timerText.innerText = "0.0s";

          mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
          mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
            userAudioPreview.src = URL.createObjectURL(recordedBlob);
            userAudioPreview.classList.remove('hidden');
          };

          mediaRecorder.start(100); // Coleta a cada 100ms
          recordBtn.className = "flex-1 py-3 px-4 bg-red-600 hover:bg-red-500 text-white font-bold rounded-lg animate-pulse flex items-center justify-center gap-2";
          recordText.innerText = "Parar Gravação";
          recordingInterval = setInterval(() => {
            recordingSeconds += 0.1;
            timerText.innerText = recordingSeconds.toFixed(1) + "s";
          }, 100);
        } catch (err) {
          alert("Não foi possível acessar o microfone: " + err.message);
        }
      }
    });

    // Comparar
    const compareBtn = document.getElementById('compareBtn');
    compareBtn.addEventListener('click', async () => {
      const refId = document.getElementById('clipSelect').value;
      const fileInput = document.getElementById('fileUpload');
      let userAudio = recordedBlob;

      if (!userAudio && fileInput.files.length > 0) {
        userAudio = fileInput.files[0];
      }

      if (!refId) {
        alert("Selecione um áudio de referência!");
        return;
      }
      if (!userAudio) {
        alert("Grave pelo microfone ou selecione um arquivo de áudio para sua tentativa!");
        return;
      }

      compareBtn.disabled = true;
      compareBtn.innerText = "⏳ A Rede Neural está analisando...";

      const formData = new FormData();
      formData.append('ref_id', refId);
      formData.append('user_file', userAudio, 'attempt.wav');

      try {
        const res = await fetch('/api/compare', { method: 'POST', body: formData });
        const data = await res.json();
        
        document.getElementById('resultsCard').classList.remove('hidden');
        document.getElementById('totalScore').innerText = data.total_score;
        document.getElementById('neuralScore').innerText = data.neural_score + "%";
        document.getElementById('pitchScore').innerText = data.pitch_score + "%";
        document.getElementById('rhythmScore').innerText = data.rhythm_score + "%";
        document.getElementById('feedbackBox').innerText = (data.feedback && data.feedback.message) ? data.feedback.message : data.feedback;

        drawPitchCurves(data.visualizations.pitch_ref, data.visualizations.pitch_usr);
      } catch (err) {
        alert("Erro na análise: " + err.message);
      } finally {
        compareBtn.disabled = false;
        compareBtn.innerText = "⚡ Comparar Tentativa com a Rede Neural";
      }
    });

    function drawPitchCurves(ref, usr) {
      const canvas = document.getElementById('pitchCanvas');
      const ctx = canvas.getContext('2d');
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = 140;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const maxVal = Math.max(...ref, ...usr, 300);

      function drawLine(points, color) {
        if (!points || points.length === 0) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        points.forEach((p, i) => {
          const x = (i / (points.length - 1)) * canvas.width;
          const y = canvas.height - ((p / maxVal) * (canvas.height - 20) + 10);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      drawLine(ref, '#22d3ee'); // ciano
      drawLine(usr, '#c084fc'); // roxo
    }

    initClips();
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
