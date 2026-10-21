# 🎤 VoiceParty — Core AI & Neural Network Engine

Este é o **núcleo funcional de Inteligência Artificial** do VoiceParty. O sistema compara a imitação de voz gravada pelo usuário com o áudio original de memes e falas de filmes famosos, combinando:

1. **Rede Neural Siamesa (CRNN em PyTorch):** Extrai embeddings de 128 dimensões a partir de espectrogramas Log-Mel e calcula a similaridade acústica de **timbre, expressão e dinâmica vocal** via produto escalar/cosseno.
2. **Entonação & Melodia Vocal (Pitch $F_0$ via DTW):** Avalia a curva de entonação (subidas e descidas de tom), normalizada para não penalizar diferenças anatômicas entre homens e mulheres.
3. **Ritmo & Cadência (Envelope RMS):** Compara a dinâmica de tempo, pausas e ênfase nas sílabas.
4. **Verificação de Frases:** Validação de palavras via distância fonética/Levenshtein.

---

## 📁 Estrutura de Arquivos Criada

- `backend/ml/model.py`: Arquitetura da **Rede Neural Siamesa (CRNN)** em PyTorch (`Conv2D` + `BatchNorm` + `BiGRU` + projeção linear L2).
- `backend/ml/dataset.py`: Gerador de tripletos (Âncora, Positivo e Negativo) com *Data Augmentation* ultrarrápido (pitch shift, time stretch, ruído e roll).
- `backend/ml/train.py`: Pipeline de treinamento da rede neural com **Triplet Margin Loss** e geração do gráfico de perda.
- `backend/ml/checkpoints/`: Contém os pesos treinados da rede (`siamese_crnn.pt`) e o gráfico da curva de perda (`loss_curve.png`).
- `backend/core/audio_processor.py`: Extração de espectrogramas Log-Mel, curvas de Pitch ($F_0$) e envelope de energia RMS.
- `backend/core/scorer.py`: Motor de pontuação ponderada (0 a 100) com geração de diagnósticos e feedback narrativo.
- `backend/cli_test.py`: Interface de linha de comando para testar comparações de áudios e treinar a rede.
- `backend/app/main.py`: Servidor FastAPI com API REST e uma **bancada de testes web interativa** (com gravação de microfone no navegador e gráfico das curvas de voz em tempo real).
- `backend/data/clips/`: Banco de áudios de exemplo (memes/filmes) + arquivo `metadata.json`.

---

## 🚀 Como Testar e Usar

O ambiente virtual Python (`.venv`) já está configurado com PyTorch, Librosa, FastAPI e demais dependências.

### Opção 1: Teste Rápido via Linha de Comando (CLI)
Para rodar um teste comparando uma boa imitação vs uma má imitação do meme *"Birl"*:
```powershell
.\.venv\Scripts\python.exe -m backend.cli_test test-demo
```

Para comparar quaisquer dois arquivos `.wav` ou `.mp3` do seu computador:
```powershell
.\.venv\Scripts\python.exe -m backend.cli_test compare caminho/do/original.wav caminho/da/sua_tentativa.wav
```

Para retreinar a Rede Neural Siamesa com novas épocas:
```powershell
.\.venv\Scripts\python.exe -m backend.cli_test train --epochs 15
```

---

### Opção 2: Iniciar a Bancada de Testes Web (Recomendado)
Para testar no seu navegador com gravação direta do microfone:

1. Inicie o servidor:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
   ```
2. Abra no navegador: [http://localhost:8000](http://localhost:8000)
3. Na página você poderá:
   - Escolher um meme/fala da lista e escutar o áudio de referência;
   - Gravar sua voz pelo microfone (ou escolher um arquivo de áudio);
   - Clicar em **"Comparar Tentativa com a Rede Neural"** e ver as 4 notas, o feedback e o gráfico das curvas de pitch sobrepostas!
