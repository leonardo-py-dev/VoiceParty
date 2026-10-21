@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo ============================================================
echo   Iniciando VoiceParty - Bancada de Testes da IA
echo ============================================================
echo Servidor rodando em: http://localhost:8000
echo Pressione CTRL+C para encerrar o servidor.
echo.
start http://localhost:8000
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
pause
