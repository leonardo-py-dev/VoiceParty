@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ============================================================
echo   VoiceParty - iniciando (1 clique)
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente virtual e instalando dependencias...
  python -m venv .venv || (echo ERRO: Python nao encontrado. Instale Python 3.10-3.12. & pause & exit /b 1)
  .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt || (echo ERRO ao instalar dependencias. & pause & exit /b 1)
)
echo Servidor em: http://localhost:8000  (CTRL+C para encerrar)
start "" http://localhost:8000
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
pause
