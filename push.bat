@echo off
cd /d "%~dp0"
echo ============================================================
echo   Enviando commits para https://github.com/leonardo-py-dev/VoiceParty
echo ============================================================
echo.
git push -u origin main
echo.
pause
