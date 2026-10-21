import os
import sys
import json
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import imageio_ffmpeg

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.core.scorer import VoiceScorer
from backend.ml.train import train_siamese_network
from backend.core.audio_processor import AudioProcessor
from backend.ml.dataset import convert_to_wav, load_and_preprocess_audio

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

# Instância compartilhada do avaliador
scorer = VoiceScorer()

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
    Endpoint principal: compara o áudio de referência com a gravação do usuário
    usando a Rede Neural Siamesa + Pitch DTW + Envelope RMS.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Determina caminho do áudio de referência
        if ref_id:
            ref_path = os.path.join(CLIPS_DIR, f"{ref_id}.wav")
            if not os.path.exists(ref_path):
                # Procura por nome do arquivo em metadata
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    clips = json.load(f)
                matched = next((c for c in clips if c["id"] == ref_id), None)
                if matched:
                    ref_path = os.path.join(CLIPS_DIR, matched["file"])
                else:
                    raise HTTPException(status_code=404, detail=f"Áudio de referência '{ref_id}' não encontrado")
        elif ref_file:
            ref_path = os.path.join(temp_dir, f"ref_{ref_file.filename}")
            with open(ref_path, "wb") as buf:
                shutil.copyfileobj(ref_file.file, buf)
        else:
            raise HTTPException(status_code=400, detail="Forneça ref_id ou ref_file")

        # 2. Salva arquivo original enviado pelo navegador e converte para WAV padronizado
        raw_user_path = os.path.join(temp_dir, f"raw_{user_file.filename}")
        with open(raw_user_path, "wb") as buf:
            shutil.copyfileobj(user_file.file, buf)

        user_wav_path = os.path.join(temp_dir, "user_clean.wav")
        if not convert_to_wav(raw_user_path, user_wav_path):
            user_wav_path = raw_user_path

        # 3. Executa a avaliação completa da IA
        result = scorer.evaluate_imitation(
            ref_audio_path=ref_path,
            user_audio_path=user_wav_path,
            target_text=target_phrase,
            user_text=user_phrase
        )

        # 4. Dados para visualização gráfica das curvas de entonação (Pitch)
        y_ref = scorer.processor.extract_pitch_curve(load_and_preprocess_audio(ref_path))[0]
        y_usr = scorer.processor.extract_pitch_curve(load_and_preprocess_audio(user_wav_path))[0]

        # Downsample para envio leve ao frontend
        step = max(1, len(y_ref) // 60)
        pitch_ref = [round(float(v), 1) for v in y_ref[::step]]
        pitch_usr = [round(float(v), 1) for v in y_usr[::step]]

        return {
            "total_score": result.total_score,
            "neural_score": result.neural_score,
            "pitch_score": result.pitch_score,
            "rhythm_score": result.rhythm_score,
            "word_score": result.word_score,
            "feedback": result.feedback,
            "details": result.details,
            "visualizations": {
                "pitch_ref": pitch_ref,
                "pitch_usr": pitch_usr
            }
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/", response_class=HTMLResponse)
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
        document.getElementById('feedbackBox').innerText = data.feedback;

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
