@echo off
setlocal
cd /d "%~dp0"
python frontend\server.py --open
endlocal
